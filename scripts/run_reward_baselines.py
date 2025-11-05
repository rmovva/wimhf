#!/usr/bin/env python3
"""Run embedding and SAE-based reward baselines for one or more datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from wimhf.reward_modeling import (
    EmbeddingBaselineConfig,
    RewardBaselineResult,
    RewardDatasetConfig,
    RewardModelConfig,
    run_reward_baselines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reward baselines on preference datasets")
    parser.add_argument("config", type=str, help="Path to JSON config listing dataset entries.")
    parser.add_argument("--output", type=str, default="reward_baselines.jsonl", help="Where to write results (JSONL).")
    parser.add_argument("--train-reward-model", action="store_true", help="Train reward models as well as compute baselines.")
    return parser.parse_args()


def load_dataset_configs(path: str) -> list[RewardDatasetConfig]:
    raw = json.loads(Path(path).read_text())
    return [RewardDatasetConfig(**entry) for entry in raw]


def main() -> None:
    args = parse_args()
    dataset_cfgs = load_dataset_configs(args.config)
    emb_cfg = EmbeddingBaselineConfig()
    rm_cfg = RewardModelConfig() if args.train_reward_model else None

    results: list[RewardBaselineResult] = []
    for cfg in dataset_cfgs:
        print(f"Running reward baselines for {cfg.name}...")
        result = run_reward_baselines(cfg, emb_cfg, rm_cfg)
        results.append(result)

    df = pd.DataFrame([result.__dict__ for result in results])
    df.to_json(args.output, orient="records", lines=True)
    print(f"Wrote baseline metrics to {args.output}")

if __name__ == "__main__":
    main()
