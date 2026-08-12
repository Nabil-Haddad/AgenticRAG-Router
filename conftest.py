import sys
from pathlib import Path

# ensures `import src...` resolves regardless of how pytest is invoked
# (bare `pytest`, `python -m pytest`, or from a different cwd)
sys.path.insert(0, str(Path(__file__).resolve().parent))
