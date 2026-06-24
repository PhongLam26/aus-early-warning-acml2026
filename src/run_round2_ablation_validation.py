from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "08_run_round2_ablation_validation.py"
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
