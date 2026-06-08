from __future__ import annotations

import argparse

import pandas as pd

from utils import add_common_args, prompt_bank_path, read_csv, write_csv, write_json


def split_column(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def normalize_prompt_bank(dataset: str, root) -> pd.DataFrame:
    df = read_csv(prompt_bank_path(dataset, root))
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for column in ("keywords", "endings", "labels"):
        if column in df.columns:
            df[column] = df[column].apply(lambda x: ",".join(split_column(x)))
    if "topic_no" not in df.columns:
        df["topic_no"] = pd.NA
    if "reverse_code" not in df.columns:
        df["reverse_code"] = pd.NA
    df.insert(0, "dataset", dataset)
    return df


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Normalize prompt and label banks."))
    args = parser.parse_args()

    covid = normalize_prompt_bank("covid", args.project_root)
    generation_covid = normalize_prompt_bank("generation_covid", args.project_root)
    validation = normalize_prompt_bank("validation", args.project_root)
    labels = read_csv(prompt_bank_path("labels", args.project_root))

    out_dir = args.project_root / "data" / "processed"
    write_csv(covid, out_dir / "covid_prompts.csv")
    write_csv(generation_covid, out_dir / "generation_covid_prompts.csv")
    write_csv(validation, out_dir / "validation_prompts.csv")
    write_csv(labels, out_dir / "projection_labels.csv")
    write_json(
        {
            "covid_rows": int(len(covid)),
            "generation_covid_rows": int(len(generation_covid)),
            "validation_rows": int(len(validation)),
            "label_rows": int(len(labels)),
        },
        out_dir / "prompt_table_summary.json",
    )


if __name__ == "__main__":
    main()
