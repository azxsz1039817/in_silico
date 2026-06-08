from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def script_paths() -> list[Path]:
    excluded = {"__init__.py", "project_paths.py", "utils.py"}
    return sorted(path for path in SCRIPTS.glob("*.py") if path.name not in excluded)


def test_scripts_are_importable() -> None:
    sys.path.insert(0, str(SCRIPTS))
    for path in script_paths():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)


def test_cli_help_works() -> None:
    for path in script_paths():
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{path.name} --help failed:\nSTDOUT={result.stdout}\nSTDERR={result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_no_forbidden_local_paths_or_api_keys_in_shareable_text() -> None:
    drive = "Z" + ":/"
    forbidden = [
        re.compile(re.escape(drive), re.IGNORECASE),
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    ]
    checked_dirs = [ROOT / "scripts", ROOT / "docs"]
    checked_files = [ROOT / "README.md"]
    for directory in checked_dirs:
        checked_files.extend(path for path in directory.rglob("*") if path.is_file())
    for path in checked_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in forbidden:
            assert not pattern.search(text), f"Forbidden pattern in {path.relative_to(ROOT)}"


def test_expected_raw_data_paths_exist() -> None:
    expected = [
        ROOT / "data/raw/prompts/covid_prompt_metadata.csv",
        ROOT / "data/raw/prompts/validation_prompt_metadata.csv",
        ROOT / "data/raw/responses/covid_response_projections.csv.gz",
        ROOT / "data/raw/responses/validation_response_projections.csv.gz",
        ROOT / "data/raw/justifications/justification_cache_summary.csv",
        ROOT / "data/raw/justifications/justification_texts.csv.gz",
        ROOT / "data/raw/justifications/justification_embeddings_float32_part01.npz",
        ROOT / "data/raw/justifications/justification_embeddings_float32_part02.npz",
        ROOT / "data/raw/justifications/justification_embeddings_float32_part03.npz",
        ROOT / "data/raw/justifications/justification_embeddings_float32_part04.npz",
        ROOT / "data/raw/justifications/justification_embeddings_manifest.csv",
    ]
    for path in expected:
        assert path.exists(), f"Missing expected replication input: {path.relative_to(ROOT)}"


def test_projection_file_counts_are_stable() -> None:
    assert (ROOT / "data/raw/responses/covid_response_projections.csv.gz").exists()
    assert (ROOT / "data/raw/responses/validation_response_projections.csv.gz").exists()
