from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from ollama import embed
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login

CREATED_AT = "created_at"
ID = "external_id"
TEXT = "clean_content"

HF_TOKEN = os.getenv("HF_HUB")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSIONS = 256
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
DF = pd.DataFrame()
def load_parquet(parquet_path: str):
    global DF
    DF = pd.read_parquet(parquet_path, columns=[ID, TEXT, CREATED_AT])
    DF[TEXT] = DF[TEXT].astype(str)
    DF = DF[DF[TEXT].str.strip() != ""]
    DF[ID] = DF[ID].astype(str)

    print(f"Loaded {len(DF)} rows from {parquet_path}")

def load_month_from_parquet(
    month: str,
    parquet: pd.DataFrame = DF,
) -> pd.DataFrame:
    global DF
    parquet = DF if parquet.empty else parquet

    # Parse month boundaries
    start = pd.Timestamp(f"2024-{month}-01T00:00:00Z")
    end  = pd.Timestamp(f"2024-{int(month) + 1:02d}-01T00:00:00Z")

    # Parse timestamps
    created_at_ts = pd.to_datetime(parquet[CREATED_AT], errors="coerce", utc=True)
    mask = created_at_ts.notna() & (created_at_ts >= start) & (created_at_ts < end)

    # Filter/clean the data, select columns
    return parquet.loc[mask, :]


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
    batch_size: int = 512,
    dimensions: int = EMBEDDING_DIMENSIONS
) -> np.ndarray:
    import time

    all_vecs = []
    n = len(texts)
    print(f"Rows to process: {n}")
    batch_times = []
    start_total = time.perf_counter()

    for i in range(0, n, batch_size):
        chunk = texts[i:i+batch_size]

        t0 = time.perf_counter()
        resp = embed(model=EMBEDDING_MODEL, input=chunk, dimensions=dimensions)
        batch_times.append((time.perf_counter() - t0))

        all_vecs.append(np.asarray(resp["embeddings"], dtype=np.float32))
        avg_ms = sum(batch_times) / len(batch_times)

        print(f"\rRows processed: {i + batch_size:,}/{n:,} | Avg time: {avg_ms:.3f} s", end="", flush=True,)

    total_ms = (time.perf_counter() - start_total)
    print(f"\nDone in {total_ms/1000:.2f}s")

    return np.vstack(all_vecs)

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

    plt.show()


# ----------------------------
# 4) KMeans clustering + output CSV with one column per cluster of external_ids
# ----------------------------
def cluster_and_export_ids_csv(
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
    ids = df_month[ID].astype(str).to_numpy()
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
    text_col: str = TEXT,
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
    text_col: str = TEXT,
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
    PARQUET_PATH = "./TS24_cleaned.parquet"
    load_parquet(PARQUET_PATH)

    for month in ["06", "07", "08", "09", "10", "11"]:
        print("-----------------------------")
        print(f"Processing month {month}...")
        df_month = load_month_from_parquet(month)

        # TESTING !
        # df_month = df_month.sample(n=2048, random_state=42).reset_index(drop=True)

        embeddings = embed_texts_ollama(df_month[TEXT].tolist())
        plot_elbow(embeddings, k_max=30, save_path=f"./elbow_{month}.png")


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
