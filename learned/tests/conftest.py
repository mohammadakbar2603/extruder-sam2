"""Put learned/src (learned_pathway) and src (extruder) on the path for these tests."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "learned" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
