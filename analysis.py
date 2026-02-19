from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np


# --------- FILE PATHS ----------
TRUTH_SOCIAL_PARQUET = "/Users/destroyerofworlds/Desktop/Research/Twitter2024/TS24_cleaned.parquet"
BLUESKY_PARQUET      = "/Users/destroyerofworlds/Desktop/Research/Twitter2024/bsky_scrape/BS24.parquet"
TWITTER_FOLDER = "/Users/destroyerofworlds/Desktop/Research/Twitter2024/X_parquets_clean"  # contains multiple .parquet
# --------- PARQUETS ------------
# ts_df = pd.read_parquet(TRUTH_SOCIAL_PARQUET) #.sample(n=100, random_state=42)
# bs_df = pd.read_parquet(BLUESKY_PARQUET) #.sample(n=100, random_state=42)
# x_df = pd.read_parquet(TWITTER_FOLDER, )

CREATED_AT = "created_at"
START = pd.Timestamp("2024-05-01T00:00:00Z")
END   = pd.Timestamp("2024-12-01T00:00:00Z")  # exclusive (covers May..Nov)
MONTHS = ["2024-05","2024-06","2024-07","2024-08","2024-09","2024-10","2024-11"]

# ------- CLEANED keywords ------------ (cleaned using clean_keywords)
X_KEYWORDS = ['2024 Elections', '2024 Presidential Election', 'Biden', 'Biden2024', 'CPAC', 'Cornel West', 'DNC', 'Dean Phillips', 'Democratic party', 'Donald Trump', 'GOP', 'Green Party', 'Independent Party', 'Jill Stein', 'Joe Biden', 'Joe Biden and Kamala Harris', 'Joseph Biden', 'KAG', 'Kamala Harris', 'MAGA', 'Marianne Williamson', 'Nikki Haley', 'No Labels', 'RFK Jr', 'RNC', 'Republican party', 'Robert F. Kennedy Jr.', 'Ron DeSantis', 'Snowballing', 'Third Party', 'Trump2024', 'US Elections', 'Vivek Ramaswamy', 'bidenharris2024', 'conservative', 'letsgobrandon', 'makeamericagreatagain', 'phillips2024', 'thedemocrats', 'trumpsupporters', 'trumptrain', 'ultramaga', 'voteblue2024', 'williamson2024']
TS_KEYWORDS = ['2024Elections', '2024PresidentialElections', '2024USElections', 'Biden', 'Biden2024', 'CPAC', 'CornellWest', 'DNC', 'Democraticparty', 'DonaldTrump', 'GOP', 'GreenParty', 'IndependentParty', 'JillStein', 'JoeBiden', 'JosephBiden', 'KAG', 'KamalaHarris', 'MAGA', 'MarianneWilliamson', 'DeanPhillips', 'NikkiHaley', 'NoLabels', 'RFKJr', 'RNC', 'Republicanparty', 'RobertF.KennedyJr.', 'RonDeSantis', 'ThirdParty', 'Trump2024', 'USElections', 'VivekRamaswamy', 'bidenharris2024', 'conservative', 'democratsoftiktok', 'letsgobrandon', 'makeamericagreatagain', 'phillips2024', 'republicansoftiktok', 'thedemocrats', 'trumpsupporters', 'trumptrain', 'ultramaga', 'voteblue2024', 'williamson2024']

def clean_keywords():
    # Normalize by removing spaces
    set1 = set(sorted([s.replace(" ", "").lower() for s in X_KEYWORDS]))
    set2 = set(sorted([s.replace(" ", "").lower() for s in TS_KEYWORDS]))

    print("\n=== X keywords ===")
    print(set1)
    print("\n=== TS keywords ===")
    print(set2)

    print("\n=== Intersection (space-removed comparison) ===")
    print(sorted(set1 & set2))
    print("\n=== Unique to List 1 ===")
    print(sorted(set1 - set2))
    print("\n=== Unique to List 2 ===")
    print(sorted(set2 - set1))
clean_keywords()


def month_counts_from_parquets(parquet_paths):
    """
    Efficient: only reads created_at column, returns counts per month (Period 'M').
    """
    counts = {}

    for p in parquet_paths:
        df = pd.read_parquet(p, columns=[CREATED_AT])
        ts = pd.to_datetime(df[CREATED_AT], errors="coerce", utc=True)
        ts = ts[(ts >= START) & (ts < END)]
        # group by month
        m = ts.dt.to_period("M").value_counts()
        for k, v in m.items():
            counts[k] = counts.get(k, 0) + int(v)

    s = pd.Series(counts)
    s.index = s.index.astype(str)  # "YYYY-MM"
    s = s.reindex(MONTHS, fill_value=0)
    return s


def posts_per_month():
    def percent_share_per_month(month_counts: pd.Series) -> pd.Series:
        total = month_counts.sum()
        return (month_counts / total) * 100.0

    def annotate_counts(x, y_pct, counts, y_offset=0.15):
        for xi, yi, c in zip(x, y_pct, counts):
            plt.text(xi, yi + y_offset, f"{int(c):,}", ha="center", va="bottom", fontsize=8)

    ts_counts = month_counts_from_parquets([TRUTH_SOCIAL_PARQUET])
    bs_counts = month_counts_from_parquets([BLUESKY_PARQUET])
    tw_counts = month_counts_from_parquets(sorted(Path(TWITTER_FOLDER).glob("*.parquet")))

    ts_pct = percent_share_per_month(ts_counts)
    bs_pct = percent_share_per_month(bs_counts)
    tw_pct = percent_share_per_month(tw_counts)

    # Plot
    x = list(MONTHS)

    plt.figure()
    plt.plot(x, ts_pct.values, marker="o", label="Truth Social")
    annotate_counts(x, ts_pct.values, ts_counts.values)

    plt.plot(x, bs_pct.values, marker="o", label="Bluesky")
    annotate_counts(x, bs_pct.values, bs_counts.values)

    plt.plot(x, tw_pct.values, marker="o", label="Twitter")
    annotate_counts(x, tw_pct.values, tw_counts.values)

    plt.xticks(rotation=45)
    plt.xlabel("Month (2024)")
    plt.ylabel("% of platform posts (May–Nov total)")
    plt.title("Monthly share of posts by platform (labels = # posts)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def monthly_word_freq(df, text_col="clean_content"):
    df["month"] = pd.to_datetime(df["created_at"], utc=True).dt.to_period("M").astype(str)
    df = df[df["month"].isin(MONTHS)]

    vec = CountVectorizer(stop_words="english", min_df=20)
    X = vec.fit_transform(df[text_col])
    vocab = vec.get_feature_names_out()

    bow = pd.DataFrame(X.toarray(), columns=vocab)
    bow["month"] = df["month"].values

    monthly = bow.groupby("month").sum()
    monthly = monthly.reindex(MONTHS, fill_value=0)

    # normalize by total words per month
    monthly = monthly.div(monthly.sum(axis=1), axis=0)

    return monthly

def plot_top_ten():
    ts_monthly = monthly_word_freq(ts_df)
    bs_monthly = monthly_word_freq(bs_df)
    # x_monthly  = monthly_word_freq(x_df)

    all_words = set(ts_monthly.columns) | set(bs_monthly.columns) # | set(x_monthly.columns)

    ts_monthly = ts_monthly.reindex(columns=all_words, fill_value=0)
    bs_monthly = bs_monthly.reindex(columns=all_words, fill_value=0)
    # x_monthly  = x_monthly.reindex(columns=all_words, fill_value=0)

    global_totals = ts_monthly.sum() + bs_monthly.sum() # + x_monthly.sum()
    top_words = global_totals.sort_values(ascending=False).head(10).index

    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=True)
    axes = axes.flatten()
    for i, word in enumerate(top_words):
        ax = axes[i]

        ax.plot(MONTHS, np.log1p(ts_monthly[word]), label="TS", color="tab:red")
        ax.plot(MONTHS, np.log1p(bs_monthly[word]), label="BS", color="tab:blue")
        # ax.plot(MONTHS, np.log1p(x_monthly[word]), label="X")

        ax.set_title(word)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.set_ylabel("log(freq)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# plot_top_ten()
# clean_keywords()
