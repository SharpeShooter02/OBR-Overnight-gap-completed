from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
for name in ('BacktestingGaps', 'orb-live-trading'):
    sys.path.insert(0, str(ROOT / name))
