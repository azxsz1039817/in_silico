from __future__ import annotations

import argparse

import pandas as pd

from utils import add_common_args, read_projection_files, write_csv, write_json


def assemble(dataset: str, root) -> pd.DataFrame:
    projections = read_projection_files(dataset, root)
    if {"named_prime", "issue_no_key", "full_prompt"}.issubset(projections.columns):
        return projections

    raise ValueError(
        f"{dataset} response input is missing required compact columns. "
        "Expected named_prime, issue_no_key, and full_prompt."
    )


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Assemble GPT-3 completion projections into long-form datasets."))
    parser.add_argument("--dataset", choices=["covid", "validation"], required=True)
    args = parser.parse_args()

    df = assemble(args.dataset, args.project_root)
    out_dir = args.project_root / "data" / "processed"
    write_csv(df, out_dir / f"{args.dataset}_responses.csv")
    write_csv(df[df["named_prime"].eq(1)], out_dir / f"{args.dataset}_responses_named.csv")
    write_json(
        {
            "dataset": args.dataset,
            "rows": int(len(df)),
            "prompts": int(df["prompt_no"].nunique()),
            "questions": int(df["question_no"].nunique()),
            "issues": int(df["issue_no"].nunique()),
        },
        out_dir / f"{args.dataset}_responses_summary.json",
    )


if __name__ == "__main__":
    main()
