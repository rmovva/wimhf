"""Reproduce WIMHF paper experiments using the modular quickstart API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wimhf.quickstart import load_config, run_wimhf_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WIMHF dataset sweep")
    parser.add_argument("config_dir", type=str, help="Directory containing dataset config files (JSON).")
    parser.add_argument("--output-dir", type=str, default="paper_outputs", help="Where to store results.")
    return parser.parse_args()


def load_configs(config_dir: str):
    for path in sorted(Path(config_dir).glob("*.json")):
        yield path.stem, load_config(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in load_configs(args.config_dir):
        print(f"Running WIMHF pipeline for {name} ...")
        results = run_wimhf_pipeline(cfg)

        feature_path = output_dir / f"{name}_features.jsonl"
        results["feature_table"].to_json(feature_path, orient="records", lines=True)

        summary_path = output_dir / f"{name}_summary.json"
        summary = {
            "summary_metrics": results["summary_metrics"],
            "auc_metrics": results["auc_metrics"],
            "kept_neurons": results["kept_neurons"],
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved outputs for {name} to {output_dir}")

if __name__ == "__main__":
    main()
