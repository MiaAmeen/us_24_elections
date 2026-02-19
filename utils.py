# import pandas as pd
import csv
from collections import defaultdict
from glob import glob
from sre_constants import IN
import duckdb
import gzip
from pathlib import Path
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import re
from bs4 import BeautifulSoup

# Dataset paths/folders
X_dir = "/Users/destroyerofworlds/Desktop/Research/Twitter2024/X_parquets_clean/"  # contains multiple .parquet
TS = "/Users/destroyerofworlds/Desktop/Research/Twitter2024/TS24_cleaned.parquet"
BS = "/Users/destroyerofworlds/Desktop/Research/Twitter2024/bsky_scrape/BS24.parquet"

# month name to digit maps
month_map = {"may": "05", "jun": "06", "jul": "07", "aug": "08", "sept": "09", "oct": "10", "nov": "11"}

# somewhat union of keywords used for data collection (look at clean_keywords output)
KEYWORDS = ['2024elections', '2024presidentialelection', '2024presidentialelections', 'biden', 'biden2024', 'bidenharris2024', 'conservative', 'cpac', 'cornelwest', 'joebidenandkamalaharris', 'deanphillips', 'democraticparty', 'dnc', 'donaldtrump', 'gop', 'greenparty', 'independentparty', 'jillstein', 'joebiden', 'josephbiden', 'kag', 'kamalaharris', 'letsgobrandon', 'maga', 'makeamericagreatagain', 'mariannewilliamson', 'nikkihaley', 'nolabels', 'phillips2024', 'republicanparty', 'rfkjr', 'rnc', 'robertf.kennedyjr.', 'rondesantis', 'thedemocrats', 'thirdparty', 'trump2024', 'trumpsupporters', 'trumptrain', 'ultramaga', 'uselections', 'vivekramaswamy', 'voteblue2024', 'williamson2024']

def meeep():
        DATA_DIR = Path("/Users/destroyerofworlds/Desktop/Research/Twitter2024/x-24-us-election")
        common = set(["conversationId", "conversationIdStr", "date", "epoch", "hashtags", "id", "id_str","lang", "likeCount", "links", "media", "mentionedUsers", "quoteCount", "quotedTweet","rawContent", "replyCount", "retweetCount", "retweetedTweet", "text", "type", "url", "user","viewCount"]) # 23 columns
        non_common = set()

        may_july = ['', 'id', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str', 'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId', 'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location', 'cash_app_handle', 'user', 'date', '_type', 'type']
        august = ['', 'type', 'id', 'username', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str', 'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId', 'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location', 'cash_app_handle', 'user', 'in_reply_to_user_id_str.1', 'location.1', 'cash_app_handle.1', 'user.1', 'date']
        september = ['', 'type', 'id', 'username', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str', 'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId', 'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location', 'cash_app_handle', 'user', 'date']
        october = ['', 'type', 'id', 'username', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str', 'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId', 'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location', 'cash_app_handle', 'user', '0', 'date']
        october_gap = ['type', 'id', 'username', 'text', 'url', 'epoch', 'id_str', 'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId', 'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'retweetedTweet', 'quotedTweet', 'coordinates', 'inReplyToTweetId', 'inReplyToTweetIdStr', 'inReplyToUser', 'source', 'sourceUrl', 'sourceLabel', 'media', 'card', 'cashtags', 'viewCount', 'place', 'user', 'epoch_dt', 'date']
        november = ['0', 'type', 'id', 'username', 'text', 'url', 'epoch', 'media', 'retweetedTweet', 'retweetedTweetID', 'retweetedUserID', 'id_str', 'lang', 'rawContent', 'replyCount', 'retweetCount', 'likeCount', 'quoteCount', 'conversationId', 'conversationIdStr', 'hashtags', 'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'location', 'cash_app_handle', 'user', 'date']

        # define schemas
        schemas = {
                "schema_1": set(may_july), # may_july, 440 files, 32 columns
                "schema_2": set(september), # september, 140 files, 32 columns
                "schema_3": set(october), # october, 24 files, 36 columns
                "schema_4": set(august), # august, 117 files, 33 columns
                "schema_5": set(october_gap), # "october_gap", 47 files, 35 columns
                "schema_6": set(november), # november, 113 files, 32 columns
        }

        def meep():
                col_to_schemas = defaultdict(list)
                for name, cols in schemas.items():
                        diff = cols - common
                        for col in diff:   # only non-common columns
                                col_to_schemas[col].append(name)
                        # print(f"{name}: {', '.join(sorted(diff))}")
                        # print("-" * 60)

                print("\n==== Unique columns per schema ====\n")
                sorted_cols = dict(sorted(col_to_schemas.items()))
                for col, schema_list in sorted_cols.items():
                        print(f"{col}: {', '.join(sorted(schema_list))}")

                quit(0)

                groups = {name: [] for name in schemas}
                groups["unknown"] = []
                for file in DATA_DIR.glob("*.csv.gz"):
                        try:
                                with gzip.open(file, "rt", newline="", encoding="utf-8") as f:
                                        reader = csv.reader(f)
                                        header = next(reader)

                                matched = False
                                for name, schema_cols in schemas.items():
                                        if set(header) == schema_cols:
                                                groups[name].append(file.name)
                                                matched = True
                                                break
                                if not matched:
                                        groups["unknown"].append(file.name)
                                        bad_files.append((file.name, len(header), header))

                        except Exception as e:
                                bad_files.append((file.name, "ERROR", str(e)))

                print("\n==== Schema Membership ====\n")
                for name, files in groups.items():
                        print(f"{name}: {len(files)} files")
                        print(", ".join(files))
                print(f"\nChecked {len(list(DATA_DIR.glob('*.csv.gz')))} files.")
                print(f"Files with issues: {len(bad_files)}")
        

def duck_ident(name: str) -> str:
    """
    Quote an identifier for DuckDB SQL, including funky names like xyz.1 or empty string.
    """
    return '"' + name.replace('"', '""') + '"'

# def wtf_i_forgot():
#         con = duckdb.connect()  # in-memory is fine
#         con.execute("PRAGMA threads=8;")
#         con.execute("PRAGMA enable_progress_bar=true;")

#         failed = []
#         processed = 0

#         for src in sorted(DATA_DIR.glob("*.parquet")):
#         try:
#                 # Get actual columns in this parquet file
#                 desc = con.execute(
#                 f"DESCRIBE SELECT * FROM read_parquet('{src.as_posix()}')"
#                 ).fetchall()
#                 cols_set = {row[0] for row in desc}  # row[0] is column_name

#                 # Build projection: keep col if exists else NULL as col
#                 select_exprs = []
#                 for c in SELECT_COLS:
#                 if c in cols_set:
#                         select_exprs.append(duck_ident(c))
#                 else:
#                         select_exprs.append(f"NULL AS {duck_ident(c)}")

#                 select_sql = ",\n    ".join(select_exprs)
#                 dest = OUT_DIR / src.name
#                 con.execute(f"""
#                 COPY (
#                         SELECT
#                         {select_sql}
#                         FROM read_parquet('{src.as_posix()}')
#                 )
#                 TO '{dest.as_posix()}'
#                 (FORMAT PARQUET, COMPRESSION ZSTD);
#                 """)

#                 processed += 1
#                 print(f"Wrote: {dest.name}")

#         except Exception as e:
#                 failed.append((src.name, str(e)))

#         con.close()

#         print(f"\nDone. Processed: {processed} files")
#         if failed:
#         print("\nFailed files:")
#         for f, err in failed:
#                 print(f"- {f}: {err}")


def merge_excels_to_parquet(xlsx1, xlsx2, out_parquet):
    import pandas as pd

    cols = [
        "url", "external_id", "author_username", "content", "associated_tags", 
        "like_count", "is_reply", "reply_count", "Scraping Date",
        "tagged_accounts", "created_at", "keyword"
    ]

    def load(df):
        df = df[cols]
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return df


    df = pd.concat([load(xlsx1), load(xlsx2)], ignore_index=True).drop_duplicates(subset="external_id").astype(str)
    df.to_parquet(out_parquet, engine="pyarrow", compression="zstd")

# merge_excels_to_parquet(pd.read_excel('/Users/destroyerofworlds/Desktop/Research/NLP Project 2025/BlueSocial/data/new_truths-all.xlsx', sheet_name="Sheet1"), 
#                         pd.read_excel('/Users/destroyerofworlds/Desktop/Research/NLP Project 2025/BlueSocial/data/TS24.xlsx', sheet_name="TS24"), 
#                         "/Users/destroyerofworlds/Desktop/Research/Twitter2024/TS24.parquet")

def clean_text(text):
    text = str(text)
    if not text or text == "": return ""

    try:
        text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    except Exception:
        pass

#     text = re.sub(r"@\w+", "", text)
    text = re.sub(r"<emoji:\s*[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()  
    if not any(c.isalnum() for c in text):
        return ""

    return text


def is_url(s):
    try:
        u = urlparse(str(s))
        return u.scheme in ("http", "https")
    except Exception:
        return False


def clean_parquet(IN: pd.DataFrame, OUT, month):
        OUT = Path(OUT)
        RENAMES = {
                # "date": "created_at",
                # "rawContent": "content",
                # "username": "author_username",
                # "id": "external_id",
                "langs": "lang",
        }
        SINCE = pd.Timestamp(f"2024-{month}-01T00:00:00Z")
        UNTIL = pd.Timestamp(f"2024-{int(month) + 1:02d}-01T00:00:00Z")  # exclusive
        # SINCE = pd.Timestamp(f"2024-05-01T00:00:00Z")
        # UNTIL = pd.Timestamp(f"2024-12-01T00:00:00Z")  # exclusive

        df = IN # .rename(columns=RENAMES)

        # helpers
        df["created_at_ts"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        lang_series = df["lang"].fillna("").astype(str).str.lower()
        lang_ok = ((lang_series == "") | (lang_series.str.contains("en", na=False)))

        # BASIC FILTERING
        df = df[
                df["external_id"].notna() & (df["external_id"].astype(str).str.strip() != "") &
                df["url"].notna() & (df["url"].astype(str).str.strip() != "") &
                df["author_username"].notna() & (df["author_username"].astype(str).str.strip() != "") &
                df["clean_content"].notna() & (df["clean_content"].astype(str).str.strip() != "") &
                df["created_at_ts"].notna() &
                (df["created_at_ts"] >= SINCE) & (df["created_at_ts"] < UNTIL) &
                (lang_ok if isinstance(lang_ok, pd.Series) else True)
        ]
        df = df[df["url"].map(is_url)]

        # Deduplicating based on external_id, keeping the latest created_at
        df["external_id_s"] = df["external_id"].astype(str)
        df = df.sort_values(
                by=["external_id_s", "created_at_ts"],
                ascending=[True, False],
                kind="mergesort",  # stable sort
        )
        df = df.drop_duplicates(subset=["external_id_s"], keep="first")
        df.drop(columns=["external_id_s"], inplace=True)

        # reformat the timestamp
        df["created_at"] = df["created_at_ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df.drop(columns=["created_at_ts"], inplace=True)

        # Clean the actual text
        df["clean_content"] = df["content"].map(clean_text) # This line is expensive !!
        df = df[df["clean_content"].astype(str).str.strip() != ""]

        # Save da file
        df.to_parquet(Path(OUT), index=False, engine="pyarrow", compression="zstd")
        print(f"Processed: {IN.name} -> {Path(OUT).name}")

# clean_parquet("/Users/destroyerofworlds/Desktop/Research/Twitter2024/TS24_cleaned.parquet", "/Users/destroyerofworlds/Desktop/Research/Twitter2024/TS24_cleanedd.parquet", "05")


def tag_keywords(IN):
        df = pd.read_parquet(Path(IN), engine="pyarrow")

        # normalize once
        text = df["text"].str.lower().str.replace(" ", "", regex=False)

        # normalize keywords once
        pattern = "|".join(map(re.escape, KEYWORDS))

        df["keyword"] = text.str.findall(pattern).str.join(",")

        df.to_parquet(IN, index=False, engine="pyarrow", compression="zstd")
        return df

tag_keywords(X_dir + "may_cleaned.parquet")

df1 = pd.read_excel("/Users/destroyerofworlds/Desktop/Research/NLP Project 2025/BlueSocial/data/new_truths-all.xlsx", sheet_name="Sheet1")
df1 = pd.read_excel("/Users/destroyerofworlds/Desktop/Research/NLP Project 2025/BlueSocial/data/new_truths-all.xlsx", sheet_name="Sheet1")

df = pd.read_parquet(Path("/Users/destroyerofworlds/Desktop/Research/Twitter2024/TS24_cleaned.parquet"))
