from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    script = SCRIPT_DIR / "07_run_improvement.py"
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
