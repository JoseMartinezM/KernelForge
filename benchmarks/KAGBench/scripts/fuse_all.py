from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "cases"
LEDGER_PATH = ROOT / "kagbench.jsonl"

CODE_FIELDS = {
    "prompt": "prompt.md",
    "pytorch_reference": "pytorch_reference.py",
    "public_tests": "public_tests.py",
    "unit_tests": "unit_tests.py",
}


def load_case(case_dir: Path) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")

    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj.setdefault("case_dir", str(case_dir.relative_to(ROOT)))

    for field, filename in CODE_FIELDS.items():
        path = case_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        obj[field] = path.read_text(encoding="utf-8").rstrip() + "\n"

    source_path = case_dir / "source.py"
    if not source_path.exists():
        raise FileNotFoundError(f"missing vendored source {source_path}")
    obj["vendored_source"] = str(source_path.relative_to(ROOT))

    required = {"id", "source_file", "entry_file", "prompt", "pytorch_reference", "public_tests", "unit_tests", "tags"}
    missing = sorted(required - obj.keys())
    if missing:
        raise ValueError(f"{case_dir}: missing required fields: {', '.join(missing)}")
    return obj


def main() -> None:
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(path for path in CASES_DIR.iterdir() if path.is_dir())
    rows = [load_case(case_dir) for case_dir in case_dirs]

    seen: set[str] = set()
    for row in rows:
        case_id = row["id"]
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)

    with LEDGER_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {len(rows)} cases to {LEDGER_PATH.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main()
