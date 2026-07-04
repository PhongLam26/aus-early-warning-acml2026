from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = PROJECT_ROOT / "src" / "12_run_sota_baseline_suite.py"


def load_baseline_module():
    src_dir = str(PROJECT_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    spec = importlib.util.spec_from_file_location("baseline_suite", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BaselineLeakageAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_baseline_module()
        cls.panel = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "model_ready_panel_improved.csv")
        cls.comparison = pd.read_csv(PROJECT_ROOT / "outputs" / "tables" / "sota_baseline_comparison.csv")

    def test_panel_shape_and_temporal_splits(self) -> None:
        keys = ["region", "crop", "year_start"]
        unique = self.panel[keys].drop_duplicates()
        self.assertEqual(len(self.panel), 4830)
        self.assertEqual(len(unique), 966)

        split_by_key = self.panel[keys + ["split"]].drop_duplicates()
        self.assertEqual(len(split_by_key), len(unique))
        years = self.panel.groupby("split")["year_start"].agg(["min", "max"]).to_dict("index")
        self.assertLessEqual(years["train"]["max"], 2012)
        self.assertGreaterEqual(years["validation"]["min"], 2013)
        self.assertLessEqual(years["validation"]["max"], 2016)
        self.assertGreaterEqual(years["test"]["min"], 2017)

    def test_feature_regimes_exclude_targets_and_leakage(self) -> None:
        blocked = self.module.TARGET_COLUMNS | self.module.LEAKAGE_COLUMNS | self.module.META_COLUMNS
        no_yield = set(self.module.feature_columns(self.panel, "no_yield_history_weather_soil"))
        operational = set(self.module.feature_columns(self.panel, "operational_with_yield_history"))

        self.assertFalse(no_yield & blocked)
        self.assertFalse(operational & (self.module.TARGET_COLUMNS | self.module.LEAKAGE_COLUMNS))
        self.assertFalse({c for c in no_yield if c.startswith("yield_lag") or "rolling" in c or "expanding" in c})

        lag_features = {c for c in operational if c.startswith("yield_lag") or "rolling" in c or "expanding" in c}
        self.assertEqual(
            lag_features,
            {"yield_lag1_crop_region", "yield_rolling3_past_crop_region", "yield_expanding_past_mean_crop_region"},
        )

    def test_baseline_outputs_are_real_successful_runs(self) -> None:
        self.assertEqual(len(self.comparison), 120)
        self.assertTrue((self.comparison["status"] == "ok").all())
        forbidden_models = self.comparison["model"].str.contains("fallback|unavailable|MLP", case=False, regex=True)
        self.assertFalse(forbidden_models.any())

        models = set(self.comparison["model"])
        for expected in ["XGBoost", "LightGBM", "CatBoost", "GAM", "DailyWeather-GRU"]:
            self.assertIn(expected, models)

        config = json.loads((PROJECT_ROOT / "outputs" / "reports" / "baseline_suite_config.json").read_text())
        self.assertTrue(config["availability"]["torch"])
        self.assertTrue(config["availability"]["pygam"])
        self.assertIn("python_executable", config["package_versions"])


if __name__ == "__main__":
    unittest.main()
