from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CASES_DIR = ROOT / "cases"

CASE_FILES = [
    "manifest.json",
    "prompt.md",
    "pytorch_reference.py",
    "public_tests.py",
    "unit_tests.py",
]


def safe_case_dir_name(case_id: str) -> str:
    return case_id.replace("/", "__")


def import_case(tmpdir: Path) -> Path:
    for filename in CASE_FILES:
        path = tmpdir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    manifest = json.loads((tmpdir / "manifest.json").read_text(encoding="utf-8"))
    case_id = manifest["id"]
    source_file = REPO_ROOT / manifest["source_file"]
    if not source_file.exists():
        raise FileNotFoundError(f"source_file does not exist: {source_file}")

    case_dir = CASES_DIR / safe_case_dir_name(case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    for filename in CASE_FILES:
        shutil.copy2(tmpdir / filename, case_dir / filename)
    shutil.copy2(source_file, case_dir / "source.py")
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a generated KAGBench case directory.")
    parser.add_argument("tmpdir", type=Path, help="Directory containing manifest.json and case files")
    args = parser.parse_args()

    case_dir = import_case(args.tmpdir.resolve())
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fuse_all.py")], check=True)
    print(f"imported {case_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
