"""Agrega el root del proyecto al path para que `compiler` sea importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
