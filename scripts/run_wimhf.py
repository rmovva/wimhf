#!/usr/bin/env python3
"""Run the WIMHF pipeline for a given dataset configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from wimhf.quickstart import load_config, run_wimhf_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WIMHF on a preference dataset.")
    parser.add_argument("config", type=str, help="Path to a JSON configuration file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where pipeline artefacts will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results = run_wimhf_pipeline(cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_table: pd.DataFrame = results["feature_table"]
    feature_table.to_json(output_dir / "feature_table.jsonl", orient="records", lines=True)

    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w") as f:
        json.dump(results["summary_metrics"], f, indent=2)

    auc_path = output_dir / "predictive_metrics.json"
    with auc_path.open("w") as f:
        json.dump(results["auc_metrics"], f, indent=2)

    coef_path = output_dir / "lasso_coefficients.json"
    with coef_path.open("w") as f:
        json.dump(results["lasso_coef_maps"], f, indent=2)

    logit_path = output_dir / "logit_coefficients.json"
    with logit_path.open("w") as f:
        json.dump(results["logit_coef_map"], f, indent=2)

    print(f"WIMHF pipeline completed. Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
