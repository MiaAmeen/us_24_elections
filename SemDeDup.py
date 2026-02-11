from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import torch

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_HUB")
login(HF_TOKEN)

CREATED_AT = "created_at"
ID = "external_id"
TEXT = "clean_content"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_PROMPT = (
        "Instruct: Compute a representation for this sentence that captures its semantic meaning for the purpose of clustering.\n"
        "Sentence:"
    )

# ----------------------------
# 1) Load + filter a parquet by month
# ----------------------------
def load_month_from_parquet(
    parquet_path: str | Path,
    month: str # "MM"
) -> pd.DataFrame:

    parquet_path = Path(parquet_path)

    # Parse month boundaries
    start = pd.Timestamp(f"2024-{month}-01T00:00:00Z")
    end  = pd.Timestamp(f"2024-{int(month) + 1:02d}-01T00:00:00Z")
    df = pd.read_parquet(parquet_path)

    # Parse timestamps
    created_at_ts = pd.to_datetime(df[CREATED_AT], errors="coerce", utc=True)
    mask = created_at_ts.notna() & (created_at_ts >= start) & (created_at_ts < end)

    # Filter/clean the data, select columns
    out = df.loc[mask, [ID, TEXT, CREATED_AT]]
    out[TEXT] = out[TEXT].astype(str)
    out = out[out[TEXT].str.strip() != ""]
    out[ID] = out[ID].astype(str)

    print(f"Loaded {len(out)} rows for month {month} from {parquet_path.name}")
    return out


# ----------------------------
# 2) Embeddings
# ----------------------------
def build_embedding_model(
    model_name: str = EMBEDDING_MODEL,
    max_seq_length: int = 512,
):
    model = SentenceTransformer(
        model_name,
        # trust_remote_code=True,
        # model_kwargs={"torch_dtype": torch.bfloat16},
        token=HF_TOKEN
    )
    model.max_seq_length = max_seq_length
    return model


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    prompt: str = EMBEDDING_PROMPT,
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """
    Returns embeddings as float32 numpy array of shape (n, d).
    """
    embs = model.encode(
        texts,
        prompt=prompt,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    # sentence-transformers may return np.ndarray already; ensure float32 for sklearn speed
    embs = np.asarray(embs, dtype=np.float32)
    return embs


# ----------------------------
# 3) Elbow method (plot inertia vs k)
# ----------------------------
def plot_elbow(
    embeddings: np.ndarray,
    k_max: int = 30,
    random_state: int = 42,
    batch_size: int = 2048,
    max_iter: int = 200,
    n_init: int = 10,
    save_path: Optional[str | Path] = None,
) -> List[Tuple[int, float]]:
    """
    Fits MiniBatchKMeans for k in [k_min, k_max], collects inertia, plots elbow.

    Returns list of (k, inertia).
    """
    ks = list(range(2, k_max + 1))
    inertias: List[float] = []

    for k in ks:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=n_init,
        )
        km.fit(embeddings)
        inertias.append(float(km.inertia_))

    # Plot
    plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow Method (MiniBatchKMeans)")
    plt.xticks(ks)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.show()
    return list(zip(ks, inertias))


# ----------------------------
# 4) KMeans clustering + output CSV with one column per cluster of external_ids
# ----------------------------
def cluster_and_export_ids_csv(
    df_month: pd.DataFrame,
    embeddings: np.ndarray,
    k: int,
    out_csv_path: str | Path,
    id_col: str = "external_id",
    random_state: int = 42,
    batch_size: int = 2048,
    max_iter: int = 200,
    n_init: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Runs MiniBatchKMeans with k clusters, assigns labels, writes a CSV where each
    column is a cluster and rows are external_ids (ragged columns padded with "").

    Returns:
      (cluster_id_table_df, labels_array)
    """
    if len(df_month) != embeddings.shape[0]:
        raise ValueError(f"Row mismatch: df has {len(df_month)} rows but embeddings has {embeddings.shape[0]} vectors.")

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
    )
    labels = km.fit_predict(embeddings)

    # Collect ids per cluster
    ids = df_month[id_col].astype(str).to_numpy()
    clusters: Dict[int, List[str]] = {i: [] for i in range(k)}
    for _id, lab in zip(ids, labels):
        clusters[int(lab)].append(_id)

    # Build ragged -> rectangular dataframe
    max_len = max(len(v) for v in clusters.values()) if k > 0 else 0
    data = {}
    for c in range(k):
        col = clusters[c]
        if len(col) < max_len:
            col = col + [""] * (max_len - len(col))
        data[f"cluster_{c}"] = col

    out_df = pd.DataFrame(data)

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    return out_df, labels


# ----------------------------
# 5) Top words per cluster + LLM label prompt builder
# ----------------------------
def top_words_by_cluster(
    df_month: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    text_col: str = "clean_content",
    top_n: int = 30,
    min_df: int = 3,
    max_df: float = 0.9,
    stop_words: str | None = "english",
) -> Dict[int, List[Tuple[str, int]]]:
    """
    Returns {cluster_id: [(word, count), ... top_n]}.
    Uses CountVectorizer over texts in each cluster, counts are raw term frequencies.
    """
    if len(df_month) != len(labels):
        raise ValueError("df_month and labels length mismatch")

    results: Dict[int, List[Tuple[str, int]]] = {}

    for c in range(k):
        texts = df_month.loc[labels == c, text_col].astype(str).tolist()
        texts = [t for t in texts if t.strip()]
        if not texts:
            results[c] = []
            continue

        vec = CountVectorizer(
            stop_words=stop_words,
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # skip 1-letter tokens
        )
        X = vec.fit_transform(texts)
        counts = np.asarray(X.sum(axis=0)).ravel()
        vocab = np.array(vec.get_feature_names_out())

        if counts.size == 0:
            results[c] = []
            continue

        top_idx = np.argsort(counts)[::-1][:top_n]
        results[c] = [(vocab[i], int(counts[i])) for i in top_idx if counts[i] > 0]

    return results


def build_cluster_label_prompt(
    cluster_id: int,
    top_words: List[Tuple[str, int]],
    sample_posts: List[str],
    instructions: str = (
        "You are labeling a cluster of social media posts for topic modeling.\n"
        "Return a short label (2–6 words) and a 1–2 sentence description.\n"
        "Avoid political bias; label the dominant theme.\n"
    ),
) -> str:
    """
    Produces a prompt you can paste into any LLM to label a cluster.
    """
    words_str = ", ".join([f"{w}({c})" for w, c in top_words[:30]])
    samples = "\n".join([f"- {s[:280].replace('\\n',' ')}" for s in sample_posts[:10]])

    prompt = (
        f"{instructions}\n"
        f"Cluster: {cluster_id}\n\n"
        f"Top words (count): {words_str}\n\n"
        f"Sample posts:\n{samples}\n\n"
        "Output format:\n"
        "Label: <2-6 words>\n"
        "Description: <1-2 sentences>\n"
    )
    return prompt


def sample_posts_for_cluster(
    df_month: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    text_col: str = "clean_content",
    n: int = 10,
    seed: int = 42,
) -> List[str]:
    subset = df_month.loc[labels == cluster_id, text_col].astype(str)
    subset = subset[subset.str.strip() != ""]
    if subset.empty:
        return []
    return subset.sample(n=min(n, len(subset)), random_state=seed).tolist()


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    PARQUET_PATH = "./TS24_clean.parquet"
    MONTH = "05"
    OUT_ELBOW_PNG = "./elbow.png"
    OUT_CLUSTER_CSV = "./clusters_ids.csv"

    # Load da data
    df_month = load_month_from_parquet(PARQUET_PATH, MONTH)

    # Do da embeddings
    model = build_embedding_model()
    embeddings = embed_texts(model, df_month["clean_content"].tolist())
    plot_elbow(embeddings, k_max=30, save_path=OUT_ELBOW_PNG)

#     # Once you pick k...
#     k = 12
#     cluster_table, labels = cluster_and_export_ids_csv(
#         df_month, embeddings, k=k, out_csv_path=OUT_CLUSTER_CSV
#     )

#     # Top words + label prompts
#     top = top_words_by_cluster(df_month, labels, k=k, top_n=30)
#     for c in range(k):
#         samples = sample_posts_for_cluster(df_month, labels, c, n=8)
#         label_prompt = build_cluster_label_prompt(c, top[c], samples)
#         print("=" * 80)
#         print(label_prompt)
