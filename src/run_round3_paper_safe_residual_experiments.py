from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).with_name("09_run_round3_paper_safe_residual_experiments.py")
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
