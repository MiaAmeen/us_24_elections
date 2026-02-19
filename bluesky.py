"""
Fetch Bluesky posts containing given keywords over a time window and write to CSV.

Uses: app.bsky.feed.searchPosts (via atproto Python SDK)
Docs: https://docs.bsky.app/docs/api/app-bsky-feed-search-posts

Notes:
- "ALL" means: everything the Bluesky search service returns for that query/time window.
- Use an App Password (recommended), not your main password.
- since is inclusive; until is exclusive (per SDK/docs).
"""
import os
import re
import csv
import random
import time
import pandas as pd
from typing import Dict, Any, Iterable, Optional, Set
import argparse

from atproto import Client
from atproto.exceptions import AtProtocolError

from dotenv import load_dotenv
load_dotenv()

# CLIENT INFO
BLUESKY_USERNAME = os.getenv("BLUESKY_USERNAME")
BLUESKY_PASSWORD = os.getenv("BLUESKY_PASSWORD")

# SCRAPE CONFIG
DATA_DIR = "./bsky_scrape/"
SLEEP = 2

# OUTPUT FIELDS
POST_FIELDS = [
        "keyword", "uri", "cid", "indexedAt", "createdAt", "text", "langs", "author_did", "replyCount", "repostCount", "likeCount", "quoteCount", "bookmarkCount", "reply_parent_uri", "reply_parent_cid", "reply_root_uri", "reply_root_cid",
]
AUTHOR_FIELDS = [
        "did", "handle", "display_name", "created_at", "avatar", "allow_subscriptions", "pronouns", "verification"
]

# KEYWORDS TO QUERY
# KEYWORDS = [
#     r'''("2024 election*" OR 2024election* OR "us election*" OR uselection*
#         OR "2024 presidential election*" OR 2024presidentialelection*)''', # Elections (general)
#     r'''(biden OR biden2024 OR "joe biden" OR joebiden 
#         OR "joseph biden" OR josephbiden 
#         OR bidenharris2024 
#         OR ("joe biden" AND "kamala harris"))''',     # Biden / Harris
#     r'''(trump OR trump2024 OR "donald trump" OR donaldtrump 
#         OR trumpsupport* OR trumptrain)''',     # Trump
#     r'''("kamala harris" OR kamalaharris)''',     # Kamala Harris (standalone)
#     r'''("nikki haley" OR nikkihaley 
#         OR "ron desantis" OR rondesantis 
#         OR "vivek ramaswamy" OR vivekramaswamy)''',     # GOP primary candidates
#     r'''(rfk* OR "rfk jr" OR rfkjr 
#         OR "robert f kennedy" 
#         OR "robert f kennedy jr" OR robertfkennedyjr*)''',     # RFK Jr
#     r'''("jill stein" OR jillstein
#         OR "cornel west" OR cornelwest OR cornellwest)''', # Other candidates
#     r'''(cpac OR gop OR rnc OR dnc
#         OR maga OR ultramaga OR kag)''', # Political orgs / movements
#     r'''(conservative
#         OR republican* OR democrat* 
#         OR "democratic party" OR democraticparty 
#         OR "republican party" OR republicanparty
#         OR "third party" OR thirdparty
#         OR "green party" OR greenparty
#         OR "independent party" OR independentparty
#         OR "no labels" OR nolabels)''', # Parties / ideology
#     r'''(voteblue2024 OR letsgobrandon OR makeamericagreatagain)''', # Slogans / hashtags
#     r'''(republicansoftiktok OR democratsoftiktok)'''     # Platform-specific political clusters
# ]
KEYWORDS = [
    "2024 Elections","2024Elections","2024 Presidential Election","2024PresidentialElections","2024USElections","US Elections","USElections","Biden","Joe Biden","JoeBiden","Joseph Biden","JosephBiden","Biden2024","bidenharris2024","Donald Trump","DonaldTrump","Trump2024","trumpsupporters","trumptrain","conservative","republicansoftiktok","democratsoftiktok","CPAC","GOP","KAG","MAGA","ultramaga","Nikki Haley","NikkiHaley","Ron DeSantis","RonDeSantis","RNC","DNC","thedemocrats","the democrats","Democratic party","Democraticparty","Republican party","Republicanparty","Third Party","ThirdParty","Green Party","GreenParty","Independent Party","IndependentParty","No Labels","NoLabels","Kamala Harris","KamalaHarris","Joe Biden and Kamala Harris","Marianne Williamson","MarianneWilliamson","Dean Phillips","DeanPhillips","williamson2024","phillips2024","RFK Jr","RFKJr","Robert F. Kennedy Jr.","RobertF.KennedyJr.","Jill Stein","JillStein","Cornel West","CornellWest","voteblue2024","letsgobrandon","makeamericagreatagain","Vivek Ramaswamy","VivekRamaswamy","Snowballing"
]

def backoff_sleep(attempt: int) -> None:
    # exponential backoff with jitter
    base = min(2 ** attempt, 30)
    time.sleep(base + random.random())


def iter_search_posts(
    client: Client,
    query: str,
    since: str,
    until: str,
    lang: str,
    limit: int = 100,
    sort: str = "top",
    sleep_s: float = 5
) -> Iterable[Dict[str, Any]]:
    """
    Generator yielding post "views" (dict-like) from searchPosts, paginating via cursor.
    """
    cursor: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"q": query, "since": since, "until": until, "limit": limit, "lang": lang, "sort": sort}
        if cursor: params["cursor"] = cursor

        # Retry transient errors (rate limits, timeouts, etc.)
        for attempt in range(3):
            try:
                resp = client.app.bsky.feed.search_posts(params=params)
                break
            except AtProtocolError as e:
                msg = str(e).lower()
                if any(k in msg for k in ["rate", "timeout", "tempor", "429", "too many", "unavailable"]):
                    backoff_sleep(attempt)
                    continue
                raise
        else:
            raise RuntimeError("Too many retries calling search_posts")

        # resp.posts: list of PostView
        posts = resp.posts or []
        for p in posts:
            if hasattr(p, "model_dump"):
                yield p.model_dump()
            else:
                yield dict(p)

        cursor = getattr(resp, "cursor", None)
        if not cursor or len(posts) == 0: return
        time.sleep(sleep_s)


def safe_text(post: Dict[str, Any]) -> str:
    # PostView has a nested record with text at record.text
    rec = post.get("record") or {}
    txt = rec.get("text") or ""
    return txt.replace("\r", " ").replace("\n", " ").strip()


def get_reply_refs(post: Dict[str, Any]) -> Dict[str, str]:
    """
    Pull reply parent/root URIs/CIDs if present; empty strings otherwise.
    """
    rec = post.get("record") or {}
    reply = rec.get("reply") or {}
    parent = reply.get("parent") or {}
    root = reply.get("root") or {}
    return {
        "reply_parent_uri": parent.get("uri", ""),
        "reply_parent_cid": parent.get("cid", ""),
        "reply_root_uri": root.get("uri", ""),
        "reply_root_cid": root.get("cid", ""),
    }


def scrape() -> int:
    # ---- Put your keywords here ----
    client = Client()
    client.login(BLUESKY_USERNAME, BLUESKY_PASSWORD)

    seen_post_uris: Set[str] = set()
    seen_author_dids: Set[str] = set()

    with open(POSTS_OUT, "w", newline="", encoding="utf-8") as pf, open(AUTHORS_OUT, "w", newline="", encoding="utf-8") as af:
        w = csv.DictWriter(pf, fieldnames=POST_FIELDS)
        a = csv.DictWriter(af, fieldnames=AUTHOR_FIELDS)
        w.writeheader()
        a.writeheader()

        total_authors = 0
        total_posts = 0

        for kw in KEYWORDS:
        #     kw = kw.lower()
            for post in iter_search_posts(
                client=client,
                query=kw,
                since=SINCE,
                until=UNTIL,
                lang=None,
                sort="latest",
                sleep_s=SLEEP,
            ):
                uri = post.get("uri", "")
                if not uri or uri in seen_post_uris:
                    continue
                seen_post_uris.add(uri)

                author = post.get("author") or {}
                rec = post.get("record") or {}

                # --- write author row once per DID ---
                author_did = author.get("did", "")
                if author_did and author_did not in seen_author_dids:
                    seen_author_dids.add(author_did)

                    associated = author.get("associated") or {}
                    activity_sub = associated.get("activity_subscription") or {}
                    allow_subs = activity_sub.get("allow_subscriptions", "")

                    author_row = {
                        "did": author_did,
                        "handle": author.get("handle", ""),
                        "display_name": author.get("display_name", author.get("displayName", "")),
                        "created_at": author.get("created_at", author.get("createdAt", "")),
                        "avatar": author.get("avatar", ""),
                        "allow_subscriptions": allow_subs,
                        "pronouns": author.get("pronouns", ""),
                        "verification": author.get("verification", "")
                    }
                    a.writerow(author_row)
                    total_authors += 1

                # --- write post row once per uri ---
                reply_refs = get_reply_refs(post)
                post_row = {
                    "keyword": kw,
                    "uri": uri,
                    "cid": post.get("cid", ""),
                    "indexedAt": post.get("indexed_at", post.get("indexedAt", "")),
                    "createdAt": rec.get("created_at", rec.get("createdAt", "")),
                    "text": safe_text(post),
                    "langs": ",".join(rec.get("langs") or []),
                    "author_did": author_did,
                    "replyCount": post.get("reply_count", post.get("replyCount", "")),
                    "repostCount": post.get("repost_count", post.get("repostCount", "")),
                    "likeCount": post.get("like_count", post.get("likeCount", "")),
                    "quoteCount": post.get("quote_count", post.get("quoteCount", "")),
                    "bookmarkCount": post.get("bookmark_count", post.get("bookmarkCount", "")),
                    **reply_refs,
                }

                w.writerow(post_row)
                total_posts += 1
                print(f"\rPosts: {total_posts:,} | Authors: {total_authors:,}", end="", flush=True)

    print()
    print(f"Done. Wrote {total_posts} unique posts to {POSTS_OUT}")
    print(f"Done. Wrote {total_authors} unique authors to {AUTHORS_OUT}")
    return 0

def merge(pattern, out_name):
    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if re.search(pattern, f)
    ]

    if not files: raise ValueError(f"No files matching pattern {pattern} in {DATA_DIR}")
    print(f"Found {len(files)} files to merge.")
    df = (
        pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
          .drop_duplicates(subset="did")
    )
    df.to_parquet(out_name, engine="pyarrow", compression="zstd")


if __name__ == "__main__":
        # parser = argparse.ArgumentParser()
        # parser.add_argument("--month", type=str, default="07", help="Month to scrape (format: MM, e.g. 07 for July)")
        # args = parser.parse_args()
        # month = args.month.zfill(2)

        # SINCE = f"2024-{month}-01T00:00:00Z"
        # UNTIL = f"2024-{int(month) + 1:02d}-01T00:00:00Z"
        # POSTS_OUT = DATA_DIR + f"bsky_posts_{month}_2.csv"
        # AUTHORS_OUT = DATA_DIR + f"bsky_users_{month}_2.csv"
        # raise SystemExit(scrape())
        merge(r"bsky_users_.*\.parquet", DATA_DIR + "bsky_users.parquet")
