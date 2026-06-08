from __future__ import annotations

import argparse

import pandas as pd

from utils import add_common_args, write_csv


LABEL_PROMPT_TEMPLATE = """You are labeling clusters of short political justifications from a replication archive.
Read the sampled justifications and provide a short substantive label for the common theme.

Cluster {cluster}
Examples:
{examples}
"""


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Create cluster-labeling prompts from cached cluster assignments."))
    parser.add_argument("--clusters", required=True, help="Path under project root to a cluster assignment CSV.")
    parser.add_argument("--examples-per-cluster", type=int, default=20)
    parser.add_argument("--output", default="data/processed/cluster_labeling_prompts.csv")
    args = parser.parse_args()

    clusters = pd.read_csv(args.project_root / args.clusters)
    rows = []
    for cluster, group in clusters.groupby("cluster"):
        examples = "\n".join(f"- {text}" for text in group["justification"].dropna().astype(str).head(args.examples_per_cluster))
        rows.append({"cluster": cluster, "labeling_prompt": LABEL_PROMPT_TEMPLATE.format(cluster=cluster, examples=examples)})
    write_csv(pd.DataFrame(rows), args.project_root / args.output)


if __name__ == "__main__":
    main()

