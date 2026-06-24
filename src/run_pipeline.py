from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


STEPS = [
    "01_audit_data.py",
    "02_build_stage_weather_features.py",
    "03_prepare_soil_features.py",
    "04_build_targets_and_panel.py",
    "05_train_evaluate_models.py",
    "06_make_figures.py",
]


def main() -> None:
    for step in STEPS:
        script = SCRIPT_DIR / step
        print(f"\n=== Running {step} ===")
        subprocess.run([sys.executable, str(script)], check=True)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
