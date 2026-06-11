"""Add the project root to the path so `compiler` is importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
