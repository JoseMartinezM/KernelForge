"""
Semantic checker entry point for the KernelForge agent.

Reads a kernel file, runs the AST-based semantic checker, and prints
warnings as JSON to stdout. Used by the Pi agent's run_semantic_check tool.

Usage:
    uv run python scripts/run_semantic_check.py --kernel-file /tmp/kernel.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kernelforge.benchmark.semantic_checker import check_kernel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic checks on a Triton kernel.")
    parser.add_argument("--kernel-file", required=True, help="Path to the kernel Python file.")
    args = parser.parse_args()

    code = Path(args.kernel_file).read_text(encoding="utf-8")
    warnings = check_kernel(code)
    print(json.dumps({"warnings": warnings}))


if __name__ == "__main__":
    main()
