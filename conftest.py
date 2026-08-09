import sys
from pathlib import Path

# src/ modules import their siblings as top-level (e.g. `from Config import Config`),
# which only works if src/ itself is on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
