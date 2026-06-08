from __future__ import annotations

import argparse

import pandas as pd

from utils import add_common_args, prompt_bank_path, write_csv, write_json


EXPECTED_COUNTS = {"correct": 112, "wrong": 9, "no_effect": 58}

EXPECTED_DIRECTION_BY_KEYWORD = {
    "closing businesses": "liberal",
    "keeping businesses open": "liberal",
    "closing schools": "liberal",
    "keeping schools open": "liberal",
}


def classify(row: pd.Series) -> str:
    if row["direction"] == "no_effect":
        return "no_effect"
    expected = row.get("expected_direction")
    if expected in {"liberal", "conservative"}:
        return "correct" if row["direction"] == expected else "wrong"
    if row["reverse_code"] == 0 and row["direction"] == "liberal":
        return "correct"
    if row["reverse_code"] == 1 and row["direction"] == "conservative":
        return "correct"
    return "wrong"


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Validate regenerated outputs against manuscript-level consistency checks."))
    args = parser.parse_args()
    root = args.project_root

    rows = []
    for dataset, expected_rows, expected_prompts, expected_questions in [
        ("covid", 179000, 358, 179),
        ("validation", 143000, 286, 143),
    ]:
        responses = pd.read_csv(root / "data" / "processed" / f"{dataset}_responses.csv")
        rows.append(
            {
                "check": f"{dataset}_response_shape",
                "observed": f"{len(responses)} rows, {responses['prompt_no'].nunique()} prompts, {responses['question_no'].nunique()} questions",
                "expected": f"{expected_rows} rows, {expected_prompts} prompts, {expected_questions} questions",
                "status": "pass"
                if len(responses) == expected_rows
                and responses["prompt_no"].nunique() == expected_prompts
                and responses["question_no"].nunique() == expected_questions
                else "review",
            }
        )

    coefs = pd.read_csv(root / "tables" / "covid_coefficients.csv")
    meta = pd.read_csv(prompt_bank_path("covid", root), encoding="utf-8-sig")[["issue_no", "reverse_code"]].drop_duplicates()
    classified = coefs.merge(meta, on="issue_no", how="left")
    classified["expected_direction"] = classified["keyword"].str.lower().map(EXPECTED_DIRECTION_BY_KEYWORD)
    classified.loc[classified["expected_direction"].isna() & classified["reverse_code"].eq(0), "expected_direction"] = "liberal"
    classified.loc[classified["expected_direction"].isna() & classified["reverse_code"].eq(1), "expected_direction"] = "conservative"
    classified["derived_forecast_result"] = classified.apply(classify, axis=1)
    observed = classified["derived_forecast_result"].value_counts().to_dict()
    rows.append(
        {
            "check": "covid_forecast_counts",
            "observed": str({key: int(observed.get(key, 0)) for key in ["correct", "wrong", "no_effect"]}),
            "expected": str(EXPECTED_COUNTS),
            "status": "review" if observed != EXPECTED_COUNTS else "pass",
        }
    )

    out = pd.DataFrame(rows)
    write_csv(out, root / "tables" / "consistency_checks.csv")
    write_csv(classified, root / "tables" / "covid_coefficients_with_consistency_check.csv")
    write_json(
        {
            "note": (
                "Forecast correctness is evaluated at the prompt-keyword level. "
                "For most prompts, expected direction follows the issue-level reverse_code field; "
                "for business and school open/closed prompts, expected direction follows the keyword shown in the coefficient table."
            )
        },
        root / "tables" / "consistency_checks_notes.json",
    )


if __name__ == "__main__":
    main()
