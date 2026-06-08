from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils import PROJECT_ROOT, add_common_args


def run(script: str, args: list[str], cwd: Path) -> None:
    cmd = [sys.executable, str(cwd / "scripts" / script), *args]
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description="Run the cached-data replication pipeline."))
    parser.add_argument("--skip-api", action="store_true", help="Accepted for clarity; API scripts are never run by default.")
    args = parser.parse_args()
    root = args.project_root.resolve()

    run("unpack_raw_data.py", ["--project-root", str(root)], root)
    run("build_prompt_tables.py", ["--project-root", str(root)], root)
    for dataset in ("covid", "validation"):
        run("assemble_responses.py", ["--project-root", str(root), "--dataset", dataset], root)
        run("coef_plots.py", ["--project-root", str(root), "--dataset", dataset], root)
    run("stat_tests.py", ["--project-root", str(root)], root)
    run("cluster_justifications.py", ["--project-root", str(root)], root)
    run("justification_figures.py", ["--project-root", str(root)], root)
    run("validate_consistency.py", ["--project-root", str(root)], root)


if __name__ == "__main__":
    main()
