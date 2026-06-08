from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"


def project_path(*parts: str | Path) -> Path:
    return PROJECT_ROOT.joinpath(*map(Path, parts))


def ensure_dirs() -> None:
    for path in (PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("encoding", "utf-8-sig")
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Replication package root. Defaults to the parent of scripts/.",
    )
    return parser


def compact_response_path(dataset: str, root: Path = PROJECT_ROOT) -> Path:
    names = {
        "covid": "covid_response_projections.csv",
        "validation": "validation_response_projections.csv",
    }
    try:
        filename = names[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(names)}") from exc
    return root / "data" / "raw" / "responses" / filename


def prompt_bank_path(dataset: str, root: Path = PROJECT_ROOT) -> Path:
    names = {
        "covid": "covid_prompt_metadata.csv",
        "generation_covid": "covid_generation_prompt_bank.csv",
        "validation": "validation_prompt_metadata.csv",
        "labels": "projection_labels.csv",
    }
    try:
        filename = names[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt bank {dataset!r}; expected one of {sorted(names)}") from exc
    return root / "data" / "raw" / "prompts" / filename


def read_projection_files(dataset: str, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    compact = compact_response_path(dataset, root)
    if not compact.exists():
        archive = compact.with_suffix(compact.suffix + ".gz")
        if archive.exists():
            raise FileNotFoundError(f"Missing unpacked response/projection input: {compact}. Run `python scripts/unpack_raw_data.py` first.")
        raise FileNotFoundError(f"Missing compact response/projection input: {compact}")
    return read_csv(compact)


def classify_ending(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    checks = [
        ("My stance on", "My stance on..."),
        ("My personal opinion on", "My personal opinion on..."),
        ("I think", "I think..."),
        ("In my opinion", "In my opinion..."),
        ("My stance is", "My stance is..."),
        ("My personal opinion is", "My personal opinion is..."),
        ("I personally believe", "I personally believe..."),
        ("I believe that", "I believe that..."),
        ("Personally, I am", "Personally, I am..."),
    ]
    for needle, label in checks:
        if needle in text:
            return label
    return text.strip()


def dense_rank_by_order(values: Iterable[object]) -> list[int]:
    mapping: dict[object, int] = {}
    ranks: list[int] = []
    for value in values:
        key = "" if pd.isna(value) else value
        if key not in mapping:
            mapping[key] = len(mapping) + 1
        ranks.append(mapping[key])
    return ranks


def slope_and_se(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    groups = data.groupby("x")["y"]
    if set(groups.groups) != {0, 1}:
        return math.nan, math.nan, len(data)
    y0 = groups.get_group(0)
    y1 = groups.get_group(1)
    coef = float(y1.mean() - y0.mean())
    se = math.sqrt(float(y1.var(ddof=1)) / len(y1) + float(y0.var(ddof=1)) / len(y0))
    return coef, se, len(data)


def exact_binom_greater(successes: int, trials: int, p: float = 0.5) -> float:
    return float(sum(math.comb(trials, k) * p**k * (1 - p) ** (trials - k) for k in range(successes, trials + 1)))


def forbidden_text_patterns() -> list[re.Pattern[str]]:
    drive = "Z" + ":/"
    return [
        re.compile(re.escape(drive), re.IGNORECASE),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"/home/[^\\s'\"]+"),
    ]
