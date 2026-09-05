"""Train MGTM directly on DSTC10-MOD."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mgtm.training.dstc10_mod import main


if __name__ == "__main__":
    main(sys.argv[1:])
