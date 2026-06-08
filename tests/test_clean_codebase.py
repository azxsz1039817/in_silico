from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_scripts_importable() -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        for path in sorted(SCRIPTS.glob("*.py")):
            if path.name == "__init__.py":
                continue
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))


def test_cli_help() -> None:
    for path in sorted(SCRIPTS.glob("*.py")):
        if path.name in {"__init__.py", "project_paths.py", "utils.py"}:
            continue
        result = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_no_obvious_secrets_or_local_paths() -> None:
    patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile("Z" + ":/", re.IGNORECASE),
        re.compile("/home/" + "akozlo"),
    ]
    checked_suffixes = {".py", ".md", ".R", ".Rmd"}
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in checked_suffixes:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                assert not pattern.search(text), f"{path} contains forbidden pattern {pattern.pattern}"


def test_expected_raw_data_present() -> None:
    expected = [
        ROOT / "data" / "raw" / "prompts" / "covid_prompt_metadata.csv",
        ROOT / "data" / "raw" / "responses" / "covid_response_projections.csv.gz",
        ROOT / "data" / "raw" / "responses" / "validation_response_projections.csv.gz",
        ROOT / "data" / "raw" / "justifications" / "justification_cache_summary.csv",
        ROOT / "data" / "raw" / "justifications" / "justification_texts.csv.gz",
        ROOT / "data" / "raw" / "justifications" / "justification_embeddings_float32_part01.npz",
        ROOT / "data" / "raw" / "justifications" / "justification_embeddings_float32_part02.npz",
        ROOT / "data" / "raw" / "justifications" / "justification_embeddings_float32_part03.npz",
        ROOT / "data" / "raw" / "justifications" / "justification_embeddings_float32_part04.npz",
        ROOT / "data" / "raw" / "justifications" / "justification_embeddings_manifest.csv",
    ]
    for path in expected:
        assert path.exists(), path
