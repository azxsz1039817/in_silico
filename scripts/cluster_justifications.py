from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import add_common_args, write_csv, write_json


VECTOR_RE = re.compile(r"(?P<sentiment>pos|neg)_embed_(?P<prompt_no>\d+)_(?P<connector>\d+)_2_fin\.pkl$")


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_embedding_archive_slice(archive_path: Path, manifest_path: Path, prompt_no: int, connector: int, sentiment: str) -> np.ndarray:
    manifest = pd.read_csv(manifest_path)
    row = manifest[
        manifest["prompt_no"].eq(prompt_no)
        & manifest["connector"].eq(connector)
        & manifest["sentiment"].eq(sentiment)
    ]
    if row.empty:
        raise ValueError(f"No embedding slice for prompt_no={prompt_no}, connector={connector}, sentiment={sentiment!r}")
    first = row.iloc[0]
    if "archive_file" in first.index:
        archive_path = manifest_path.parent / str(first["archive_file"])
        offset = int(first["archive_offset"])
    else:
        offset = int(first["offset"])
    n = int(first["n"])
    with np.load(archive_path) as archive:
        embeddings = archive["embeddings"]
        return np.asarray(embeddings[offset : offset + n], dtype=np.float32)


def load_justification_texts(text_path: Path, prompt_no: int, connector: int, sentiment: str) -> list[str]:
    if not text_path.exists() and text_path.with_suffix(text_path.suffix + ".gz").exists():
        text_path = text_path.with_suffix(text_path.suffix + ".gz")
    texts = pd.read_csv(text_path)
    sub = texts[
        texts["prompt_no"].eq(prompt_no)
        & texts["connector"].eq(connector)
        & texts["sentiment"].eq(sentiment)
    ].sort_values("justification_no")
    if sub.empty:
        raise ValueError(f"No justification texts for prompt_no={prompt_no}, connector={connector}, sentiment={sentiment!r}")
    return sub["justification"].fillna("").astype(str).tolist()


def justification_cache_summary(cache_dir: Path, summary_file: Path | None = None) -> pd.DataFrame:
    if summary_file is not None and summary_file.exists() and not cache_dir.exists():
        return pd.read_csv(summary_file)
    rows = []
    for path in sorted(cache_dir.glob("*_embed_*_2_fin.pkl")):
        match = VECTOR_RE.match(path.name)
        if not match:
            continue
        text_path = cache_dir / f"{match.group('sentiment')}_lst_{match.group('prompt_no')}_1.pkl"
        count = None
        dim = None
        try:
            vectors = load_pickle(path)
            count = len(vectors)
            dim = len(vectors[0]) if vectors else 0
        except Exception as exc:
            rows.append({"file": path.name, "error": str(exc)})
            continue
        rows.append(
            {
                "prompt_no": int(match.group("prompt_no")),
                "connector": int(match.group("connector")),
                "sentiment": match.group("sentiment"),
                "embedding_file": path.name,
                "text_file": text_path.name if text_path.exists() else "",
                "n": count,
                "dimensions": dim,
            }
        )
    return pd.DataFrame(rows)


def cluster_one(cache_dir: Path, prompt_no: int, connector: int, sentiment: str, k: int, random_state: int) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    vectors = np.asarray(load_pickle(cache_dir / f"{sentiment}_embed_{prompt_no}_{connector}_2_fin.pkl"), dtype=float)
    texts = load_pickle(cache_dir / f"{sentiment}_lst_{prompt_no}_1.pkl")
    if len(texts) != len(vectors):
        texts = texts[: len(vectors)]
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(vectors)
    metrics = {
        "silhouette": float(silhouette_score(vectors, labels)) if k > 1 and len(vectors) > k else np.nan,
        "calinski_harabasz": float(calinski_harabasz_score(vectors, labels)) if k > 1 and len(vectors) > k else np.nan,
        "davies_bouldin": float(davies_bouldin_score(vectors, labels)) if k > 1 and len(vectors) > k else np.nan,
    }
    rows = pd.DataFrame(
        {
            "prompt_no": prompt_no,
            "connector": connector,
            "sentiment": sentiment,
            "cluster": labels,
            "justification": texts,
        }
    )
    rows.attrs["metrics"] = metrics
    return rows


def cluster_one_archive(
    archive_path: Path,
    manifest_path: Path,
    text_path: Path,
    prompt_no: int,
    connector: int,
    sentiment: str,
    k: int,
    random_state: int,
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    vectors = load_embedding_archive_slice(archive_path, manifest_path, prompt_no, connector, sentiment)
    texts = load_justification_texts(text_path, prompt_no, connector, sentiment)
    if len(texts) != len(vectors):
        texts = texts[: len(vectors)]
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(vectors)
    metrics = {
        "silhouette": float(silhouette_score(vectors, labels)) if k > 1 and len(vectors) > k else np.nan,
        "calinski_harabasz": float(calinski_harabasz_score(vectors, labels)) if k > 1 and len(vectors) > k else np.nan,
        "davies_bouldin": float(davies_bouldin_score(vectors, labels)) if k > 1 and len(vectors) > k else np.nan,
    }
    rows = pd.DataFrame(
        {
            "prompt_no": prompt_no,
            "connector": connector,
            "sentiment": sentiment,
            "cluster": labels,
            "justification": texts,
        }
    )
    rows.attrs["metrics"] = metrics
    return rows


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Summarize or cluster cached justification embeddings."))
    parser.add_argument("--cache-dir", default=None, help="Optional path to the full archived justification pickle cache.")
    parser.add_argument("--summary-file", default="data/raw/justifications/justification_cache_summary.csv")
    parser.add_argument("--embeddings-archive", default=None, help="Optional single NPZ embedding archive. By default, use archive_file entries in the manifest.")
    parser.add_argument("--embeddings-manifest", default="data/raw/justifications/justification_embeddings_manifest.csv")
    parser.add_argument("--justification-texts", default="data/raw/justifications/justification_texts.csv")
    parser.add_argument("--prompt-no", type=int)
    parser.add_argument("--connector", type=int, default=1)
    parser.add_argument("--sentiment", choices=["pos", "neg"], default="pos")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    cache_dir = args.project_root / args.cache_dir if args.cache_dir else args.project_root / "__missing_justification_pickle_cache__"
    out_dir = args.project_root / "data" / "processed"
    summary = justification_cache_summary(cache_dir, args.project_root / args.summary_file)
    write_csv(summary, out_dir / "justification_cache_summary.csv")

    if args.prompt_no is None:
        write_json({"cache_files": int(len(summary)), "mode": "summary_only"}, out_dir / "justification_cluster_summary.json")
        return

    manifest_path = args.project_root / args.embeddings_manifest
    if args.embeddings_archive:
        archive_path = args.project_root / args.embeddings_archive
    else:
        archive_path = manifest_path.parent / "single_embedding_archive.npz"
    text_path = args.project_root / args.justification_texts
    if manifest_path.exists() and (archive_path.exists() or "archive_file" in pd.read_csv(manifest_path, nrows=1).columns):
        clustered = cluster_one_archive(
            archive_path,
            manifest_path,
            text_path,
            args.prompt_no,
            args.connector,
            args.sentiment,
            args.k,
            args.random_state,
        )
    elif cache_dir.exists():
        clustered = cluster_one(cache_dir, args.prompt_no, args.connector, args.sentiment, args.k, args.random_state)
    else:
        raise FileNotFoundError(
            "No justification embedding source found. Use the included float32 NPZ archive, "
            "or provide --cache-dir pointing to the archived pickle cache."
        )
    write_csv(clustered, out_dir / f"justifications_prompt_{args.prompt_no}_{args.sentiment}_clusters.csv")
    write_json(clustered.attrs["metrics"], out_dir / f"justifications_prompt_{args.prompt_no}_{args.sentiment}_cluster_metrics.json")


if __name__ == "__main__":
    main()
