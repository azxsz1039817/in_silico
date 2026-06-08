from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

from utils import add_common_args


RAW_ARCHIVES = [
    Path("data/raw/responses/covid_response_projections.csv.gz"),
    Path("data/raw/responses/validation_response_projections.csv.gz"),
    Path("data/raw/justifications/justification_texts.csv.gz"),
]


def unpack_one(archive_path: Path, force: bool = False) -> str:
    if not archive_path.exists():
        return f"missing archive: {archive_path}"
    if archive_path.suffix != ".gz":
        return f"not a gzip archive: {archive_path}"

    output_path = archive_path.with_suffix("")
    if output_path.exists() and not force:
        return f"exists: {output_path}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive_path, "rb") as source, output_path.open("wb") as target:
        shutil.copyfileobj(source, target)
    return f"unpacked: {archive_path} -> {output_path}"


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Unpack compressed raw replication data files."))
    parser.add_argument("--force", action="store_true", help="Overwrite existing unpacked CSV files.")
    args = parser.parse_args()

    for relative_archive in RAW_ARCHIVES:
        print(unpack_one(args.project_root / relative_archive, force=args.force))


if __name__ == "__main__":
    main()
