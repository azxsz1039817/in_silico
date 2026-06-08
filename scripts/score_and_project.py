from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np
import pandas as pd

from utils import add_common_args, prompt_bank_path, read_csv, write_csv


def cosine(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    denom = math.sqrt(float(np.dot(x_arr, x_arr))) * math.sqrt(float(np.dot(y_arr, y_arr)))
    return float(np.dot(x_arr, y_arr) / denom)


def projection_score(response_embedding: Sequence[float], positive_anchor: Sequence[float], negative_anchor: Sequence[float]) -> float:
    return cosine(response_embedding, positive_anchor) - cosine(response_embedding, negative_anchor)


def main() -> None:
    parser = add_common_args(
        argparse.ArgumentParser(
            description=(
                "Document the semantic-axis scoring rule. The released replication uses cached projection files; "
                "this utility provides the cosine-difference implementation for optional embedding reruns."
            )
        )
    )
    parser.add_argument("--dataset", choices=["covid", "validation"], default="covid")
    parser.add_argument("--output", default=None, help="Optional output CSV path for prompt label pairs.")
    args = parser.parse_args()

    prompts = read_csv(prompt_bank_path(args.dataset, args.project_root))
    rows = []
    for row in prompts.itertuples(index=False):
        labels = [x.strip() for x in str(row.labels).split(",") if x.strip()]
        rows.append(
            {
                "issue_no": row.issue_no,
                "labels": ",".join(labels),
                "positive_anchor": labels[0] if labels else "",
                "negative_anchor": labels[1] if len(labels) > 1 else "",
                "scoring_rule": "cosine(response, positive_anchor) - cosine(response, negative_anchor)",
            }
        )
    out = pd.DataFrame(rows)
    if args.output:
        write_csv(out, args.project_root / args.output)
    else:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()

