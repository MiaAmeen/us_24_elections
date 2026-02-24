from __future__ import annotations

import os
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import ollama
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
from globals import *

HF_TOKEN = os.getenv("HF_HUB")
EMBEDDING_MODEL = "qwen3-embedding:8b"
EMBEDDING_DIMENSIONS = 128
EMBEDDING_PROMPT = (
        "Instruct: Compute a representation for this sentence that captures its semantic meaning for the purpose of clustering.\n"
        "Sentence:"
)
DEVICE = "cuda" if torch.cuda.is_available() \
        else "mps" if torch.backends.mps.is_available() \
        else "cpu"
print(f"Using device: {DEVICE}")

# ----------------------------
# 1) Load + filter a parquet by month
# ----------------------------
def load_parquet(parquet_path: str, cols = [ID_COL, CLEAN_TEXT_COL, CREATED_AT_COL], month: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path, columns=cols)
    df[CLEAN_TEXT_COL] = df[CLEAN_TEXT_COL].astype(str)
    df = df[df[CLEAN_TEXT_COL].str.strip() != ""]
    df[ID_COL] = df[ID_COL].astype(str)

    if month is not None:
        # Parse month boundaries
        start = pd.Timestamp(f"2024-{month}-01T00:00:00Z")
        end  = pd.Timestamp(f"2024-{int(month) + 1:02d}-01T00:00:00Z")
        created_at_ts = pd.to_datetime(df[CREATED_AT_COL], errors="coerce", utc=True)
        mask = created_at_ts.notna() & (created_at_ts >= start) & (created_at_ts < end)
        df = df.loc[mask]

    print(f"Loaded {len(df)} rows from {parquet_path}")
    return df


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
        token=HF_TOKEN,
        device=DEVICE
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
    login(HF_TOKEN)

    embs = model.encode(
        texts,
        prompt=prompt,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    # sentence-transformers may return np.ndarray already; ensure float32 for sklearn speed
    embs = np.asarray(embs, dtype=np.float32)
    return embs


def embed_texts_ollama(
    texts: List[str],
    output_path: str = "embeddings.dat",
    batch_size: int = 512,
    dimensions: int = EMBEDDING_DIMENSIONS,
):
    n = len(texts)
    print(f"Rows to process: {n}")

    # Memory-mapped array (writes directly to disk)
    embeddings = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(n, dimensions),)

    start_total = time.perf_counter()
    total_batch_time = 0.0
    batch_count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = texts[start:end]

        t0 = time.perf_counter()

        try:
            embeddings[start:end] = ollama.embed(model=EMBEDDING_MODEL, input=chunk, dimensions=dimensions)["embeddings"]
        except Exception as _:
            embeddings[start:end] = np.nan

        batch_time = time.perf_counter() - t0
        total_batch_time += batch_time
        batch_count += 1

        avg_time = total_batch_time / batch_count
        print(
            f"\rRows processed: {end:,}/{n:,} | Avg batch time: {avg_time:.3f}s",
            end="",
            flush=True,
        )

    embeddings.flush()

    total_time = time.perf_counter() - start_total
    print(f"\nDone in {total_time:.2f}s")

    return embeddings


# ----------------------------
# 3) Elbow method (plot inertia vs k)
# ----------------------------
def plot_elbow(
    embeddings: np.ndarray,
    k_max: int = 30,
    random_state: int = 42,
    batch_size: int = 2048,
    max_iter: int = 200,
    n_init: int = 20,
    save_path: Optional[str | Path] = None,
):
    """
    Fits MiniBatchKMeans for k in [k_min, k_max], collects inertia, plots elbow.

    Returns list of (k, inertia).
    """
    from sklearn.metrics import silhouette_score

    ks = list(range(2, k_max + 1))
    inertias: List[float] = []
    rng = np.random.default_rng(random_state)
    mask = rng.choice(embeddings.shape[0], 5000, replace=False)
    silhouettes = []

    for k in ks:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=n_init,
        )
        labels = km.fit_predict(embeddings)
        inertias.append(float(km.inertia_))
        silhouettes.append(silhouette_score(embeddings[mask], labels[mask]))
        # silhouettes.append(silhouette_score(embeddings, labels))

    # Plot
    _, ax1 = plt.subplots()
    ax1.plot(ks, inertias, marker="o", color="tab:blue")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (full data)", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="o", color="tab:orange")
    ax2.set_ylabel("Silhouette (sampled)", color="tab:orange")

    plt.title("Inertia + Silhouette vs k")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close()


# ----------------------------
# 4) KMeans clustering + output CSV with one column per cluster of external_ids
# ----------------------------
def kmeans_cluster(
    df_month: pd.DataFrame,
    embeddings: np.ndarray,
    k: int,
    out_csv_path: str | Path,
    random_state: int = 42,
    batch_size: int = 2048,
    max_iter: int = 200,
    n_init: int = 20,
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
    ids = df_month[ID_COL].astype(str).to_numpy()
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


def hdbscan_cluster(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    out_csv_path: str | Path,
    min_cluster_size: int = 50,
    min_samples: int = 3,
    metric: str = "cosine",
    n_jobs: int = -1,
):
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=n_jobs,
    )
    labels = clusterer.fit_predict(embeddings)

    out_df = pd.DataFrame({
        "source_TS": df["source_TS"].values,
        "id": df[ID_COL].values,
        "cluster": labels
    })

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False, chunksize=1_000_000)

    return out_df


# ----------------------------
# 5) Top words per cluster + LLM label prompt builder
# ----------------------------
def top_words_by_cluster(
    df_month: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    text_col: str = CLEAN_TEXT_COL,
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
    text_col: str = CLEAN_TEXT_COL,
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

    # # Load .npz
    # npz = np.load("embeddings_ts05.npz", allow_pickle=True)
    # embeddings = npz["embeddings"]
    # ids = npz["ids"]

    # print("EMBEDDINGS -> ", "Shape:", embeddings.shape, "Dtype:", embeddings.dtype)
    # print("IDS -> ", "Shape:", ids.shape, "Dtype:", ids.dtype)

    # # Load original dataset
    # df = load_parquet(TRUTH_SOCIAL_FILE, month="05")
    # text = df.loc[df[ID_COL] == ids[0], "clean_content"].iloc[0]

    # test_emb = embed_texts_ollama([text])
    # print("Original saved embedding matches test embedding?", np.allclose(embeddings[0], test_emb))

    # quit(0)

    for month in ["06", "07", "08", "09", "10", "11"]:
        # TRUTH SOCIAL
        ts = load_parquet(TRUTH_SOCIAL_FILE, month=month)
        ts.drop(columns=[CREATED_AT_COL], inplace=True)

        embeddings = embed_texts_ollama(ts[CLEAN_TEXT_COL].tolist(), f"embeddings.dat")
        np.savez_compressed(
            f"embeddings_ts{month}.npz",
            embeddings=embeddings,
            ids=ts[ID_COL].to_numpy()
        )
 
        # BLUESKY
        bs = load_parquet(BLUESKY_FILE, month=month)
        bs.drop(columns=[CREATED_AT_COL], inplace=True)

        embeddings = embed_texts_ollama(bs[CLEAN_TEXT_COL].tolist(), f"embeddings.dat")
        np.savez_compressed(
            f"embeddings_bs{month}.npz",
            embeddings=embeddings,
            ids=bs[ID_COL].to_numpy()
        )

        # hdbscan_cluster(df, embeddings, f"hdbscan_clusters_{month}.csv", min_cluster_size=50, min_samples=5)

