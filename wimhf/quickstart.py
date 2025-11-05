"""High-level helpers for running the WIMHF pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd

from . import embedding
from .feature_selection import (
    select_neurons_controlled_lasso,
    select_neurons_controlled_ols,
)
from .interpretation import (
    InterpretConfig,
    LLMConfig,
    NeuronInterpreter,
    SamplingConfig,
    ScoringConfig,
    sample_percentile_bins,
    sample_top_only,
)
from .llm_tasks import abbreviate_concept, parallel_apply
from .sae import SparseAutoencoder, get_sae_checkpoint_name, load_model
from .utils import (
    get_artifact_subdir,
    dedup_pair_rows,
    fit_linear_model,
    load_json,
    measure_interpretation_redundancy,
    pairwise_response_parsing_function,
    row_to_pairwise_example,
    save_json,
    create_connected_component_split,
)


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    name: str
    path: str
    split_columns: List[str]
    train_split_size: float = 0.8
    split_random_seed: int = 42
    prompt_column: str = "prompt"
    response_a_column: str = "response_A"
    response_b_column: str = "response_B"
    label_column: str = "label"
    max_words_prompt: int = 128
    max_words_response: int = 128
    random_seed: int = 42


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    n_workers: int = 2


@dataclass
class SAEConfig:
    M: int = 32
    K: int = 4
    prefix_lengths: List[int] = field(default_factory=lambda: [8, 32])
    batch_size: int = 256
    n_epochs: int = 50
    min_epochs: int = 10
    patience: int = 3
    learning_rate: float = 5e-4
    dead_neuron_threshold_steps: int = 64
    aux_k: Optional[int] = None
    multi_k: Optional[int] = None


@dataclass
class InterpretationSettings:
    interpreter_model: str = "gpt-5"
    annotator_model: str = "gpt-5-mini"
    abbreviator_model: str = "gpt-5-mini"
    n_candidates: int = 5
    interpret_n_examples: int = 5
    interpret_sampling_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"percentile_bin": (95, 100), "nonzero_only": True, "swap_negative_response_pairs": True}
    )
    scoring_n_examples: int = 300
    scoring_sampling_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"high_percentile": (50, 100), "low_percentile": (0, 50), "nonzero_only": True}
    )
    min_correlation: float = 0.0
    p_value_threshold: Optional[float] = 0.05 / 32  # Bonferroni correction based on M features
    n_workers_interpretation: int = 80
    n_workers_annotation: int = 300
    timeout: Optional[float] = None
    max_interpretation_tokens: Optional[int] = None
    completion_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionSettings:
    controls: List[str] = field(default_factory=lambda: ["length_delta"])
    lasso_top_k: List[int] = field(default_factory=lambda: [5, 10])
    max_samples_lasso: Optional[int] = None
    classification: bool = True
    standardize: bool = True
    use_lasso: bool = True


@dataclass
class WandbSettings:
    enable: bool = False
    project: Optional[str] = None
    run_name: Optional[str] = None
    entity: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    checkpoint_dir: Optional[str] = None
    retrain_sae: bool = False
    cache_dir: Optional[Path] = None
    debug: bool = False
    wandb: Optional[WandbSettings] = None


@dataclass
class WIMHFConfig:
    dataset: DatasetConfig
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    interpretation: InterpretationSettings = field(default_factory=InterpretationSettings)
    selection: SelectionSettings = field(default_factory=SelectionSettings)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, orient="records", lines=True)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def _ensure_length_delta(df: pd.DataFrame, cfg: DatasetConfig) -> None:
    if "length_delta" not in df.columns:
        a = df[cfg.response_a_column].astype(str).str.split().str.len()
        b = df[cfg.response_b_column].astype(str).str.split().str.len()
        df["length_delta"] = a - b


def _embed_unique_responses(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: DatasetConfig,
    emb_cfg: EmbeddingConfig,
    cache_dir: Optional[Path],
) -> Dict[str, np.ndarray]:
    train_texts = list(train_df[cfg.response_a_column].unique()) + list(train_df[cfg.response_b_column].unique())
    val_texts = list(val_df[cfg.response_a_column].unique()) + list(val_df[cfg.response_b_column].unique())
    unique_texts = list(dict.fromkeys(train_texts + val_texts))
    return embedding.get_openai_embeddings(
        texts=unique_texts,
        model=emb_cfg.model,
        n_workers=emb_cfg.n_workers,
        cache_dir=cache_dir,
    )


def _build_deltas(df: pd.DataFrame, cfg: DatasetConfig, response2embedding: Dict[str, np.ndarray]) -> np.ndarray:
    resp_a = np.stack([response2embedding[row[cfg.response_a_column]] for _, row in df.iterrows()])
    resp_b = np.stack([response2embedding[row[cfg.response_b_column]] for _, row in df.iterrows()])
    return resp_a - resp_b


def _augment_deltas(delta: np.ndarray) -> np.ndarray:
    return np.concatenate([delta, -delta], axis=0)


def _train_or_load_sae(
    delta_train: np.ndarray,
    delta_val: Optional[np.ndarray],
    sae_cfg: SAEConfig,
    runtime_cfg: RuntimeConfig,
) -> SparseAutoencoder:
    checkpoint_dir = runtime_cfg.checkpoint_dir
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = get_sae_checkpoint_name(sae_cfg.M, sae_cfg.K, sae_cfg.prefix_lengths)
        path = os.path.join(checkpoint_dir, filename)
        if os.path.exists(path) and not runtime_cfg.retrain_sae:
            return load_model(path)

    sae = SparseAutoencoder(
        input_dim=delta_train.shape[1],
        m_total_neurons=sae_cfg.M,
        k_active_neurons=sae_cfg.K,
        aux_k=sae_cfg.aux_k,
        multi_k=sae_cfg.multi_k,
        dead_neuron_threshold_steps=sae_cfg.dead_neuron_threshold_steps,
        prefix_lengths=sae_cfg.prefix_lengths,
    )

    sae.fit(
        X_train=torch.tensor(delta_train, dtype=torch.float),
        X_val=torch.tensor(delta_val, dtype=torch.float) if delta_val is not None else None,
        save_dir=checkpoint_dir,
        batch_size=sae_cfg.batch_size,
        learning_rate=sae_cfg.learning_rate,
        n_epochs=sae_cfg.n_epochs,
        min_epochs=sae_cfg.min_epochs,
        patience=sae_cfg.patience,
    )
    return sae


def _row_to_pairwise(row: pd.Series, cfg: DatasetConfig) -> str:
    return row_to_pairwise_example(
        row,
        include_prompt=True,
        max_words_prompt=cfg.max_words_prompt,
        max_words_response=cfg.max_words_response,
    )


def _abbreviate_interpretations(
    interpretations: List[str],
    abbreviator_model: str,
    n_workers: int = 30,
) -> List[str]:
    def _fn(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        return abbreviate_concept(stripped, model=abbreviator_model)

    return parallel_apply(interpretations, _fn, n_workers=n_workers, desc="Abbreviating")


# -----------------------------------------------------------------------------
# Public pipeline API
# -----------------------------------------------------------------------------


def load_and_preprocess_dataframe(
    dataset_cfg: DatasetConfig,
    runtime_cfg: Optional[RuntimeConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataframe and perform preprocessing steps (splitting, subsetting, dedup).
    
    Returns:
        (train_df, val_df, train_pred_df, val_pred_df, dedup_train_df, dedup_val_df)
    """
    if runtime_cfg and runtime_cfg.debug:
        df = _load_dataframe(dataset_cfg.path)
        max_rows = min(len(df), 10_000)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=dataset_cfg.random_seed).reset_index(drop=True)
    else:
        df = _load_dataframe(dataset_cfg.path)
    
    _ensure_length_delta(df, dataset_cfg)
    
    train_df, val_df = create_connected_component_split(
        df,
        dataset_cfg.split_columns,
        train_frac=dataset_cfg.train_split_size,
        seed=dataset_cfg.split_random_seed,
    )
    
    train_pred_mask = train_df[dataset_cfg.label_column].isin([0, 1]).to_numpy()
    val_pred_mask = val_df[dataset_cfg.label_column].isin([0, 1]).to_numpy()
    train_pred_df = train_df[train_pred_mask].reset_index(drop=True)
    val_pred_df = val_df[val_pred_mask].reset_index(drop=True)
    
    dedup_train_df = dedup_pair_rows(train_df, label="train")
    dedup_val_df = dedup_pair_rows(val_df, label="val")
    
    return train_df, val_df, train_pred_df, val_pred_df, dedup_train_df, dedup_val_df


def get_embeddings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dedup_train_df: pd.DataFrame,
    dedup_val_df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
    embedding_cfg: EmbeddingConfig,
    cache_dir: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get embeddings for all responses and build response deltas.
    
    Returns:
        (response2embedding, delta_train, delta_val, dedup_delta_train, dedup_delta_val)
    """
    response2embedding = _embed_unique_responses(
        train_df, val_df, dataset_cfg, embedding_cfg, cache_dir
    )
    
    delta_train = _build_deltas(train_df, dataset_cfg, response2embedding)
    delta_val = _build_deltas(val_df, dataset_cfg, response2embedding)
    dedup_delta_train = _build_deltas(dedup_train_df, dataset_cfg, response2embedding)
    dedup_delta_val = _build_deltas(dedup_val_df, dataset_cfg, response2embedding)
    
    return (
        response2embedding,
        delta_train,
        delta_val,
        dedup_delta_train,
        dedup_delta_val,
    )


def train_sae(
    dedup_delta_train: np.ndarray,
    dedup_delta_val: Optional[np.ndarray],
    sae_cfg: SAEConfig,
    runtime_cfg: RuntimeConfig,
) -> SparseAutoencoder:
    """Train or load SAE."""
    augmented_train = _augment_deltas(dedup_delta_train)
    return _train_or_load_sae(augmented_train, dedup_delta_val, sae_cfg, runtime_cfg)


def interpret_sae_features(
    dedup_train_df: pd.DataFrame,
    activ_dedup: np.ndarray,
    dataset_cfg: DatasetConfig,
    interpret_cfg: InterpretationSettings,
    runtime_cfg: RuntimeConfig,
) -> Tuple[Dict[int, List[str]], NeuronInterpreter]:
    """
    Interpret SAE features using LLMs.
    
    Returns:
        (interpretations, interpreter)
    """
    dedup_texts = dedup_train_df.apply(
        lambda row: _row_to_pairwise(row, dataset_cfg), axis=1
    ).tolist()
    
    sampling_config = SamplingConfig(
        function=sample_top_only,
        n_examples=interpret_cfg.interpret_n_examples,
        max_words_per_example=None,  # Components already truncated in row_to_pairwise_example
        sampling_kwargs=interpret_cfg.interpret_sampling_kwargs,
    )
    completion_kwargs = dict(interpret_cfg.completion_kwargs)
    if not completion_kwargs and "gpt-5" in interpret_cfg.interpreter_model:
        completion_kwargs = {"reasoning_effort": "low"}
    llm_config = LLMConfig(
        max_interpretation_tokens=interpret_cfg.max_interpretation_tokens if interpret_cfg.max_interpretation_tokens is not None else 256,
        completion_kwargs=completion_kwargs,
        timeout=interpret_cfg.timeout if interpret_cfg.timeout is not None else 30.0,
    )
    interpret_config = InterpretConfig(
        sampling=sampling_config,
        llm=llm_config,
        n_candidates=interpret_cfg.n_candidates,
    )
    
    interpreter = NeuronInterpreter(
        interpreter_model=interpret_cfg.interpreter_model,
        annotator_model=interpret_cfg.annotator_model,
        cache_dir=runtime_cfg.cache_dir,
        n_workers_interpretation=interpret_cfg.n_workers_interpretation,
        n_workers_annotation=interpret_cfg.n_workers_annotation,
    )
    
    interpretations = interpreter.interpret_neurons(
        texts=dedup_texts,
        activations=activ_dedup,
        neuron_indices=list(range(activ_dedup.shape[1])),
        config=interpret_config,
    )
    
    return interpretations, interpreter


def score_interpretations(
    interpretations: Dict[int, List[str]],
    train_df: pd.DataFrame,
    activ_train: np.ndarray,
    dataset_cfg: DatasetConfig,
    interpret_cfg: InterpretationSettings,
    interpreter: NeuronInterpreter,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[int, str], Dict[int, Dict[str, float]], List[int]]:
    """
    Score interpretations and filter by fidelity.
    
    Returns:
        (scoring_metrics, best_interpretations, best_metrics, kept_neurons)
    """
    train_texts = train_df.apply(
        lambda row: _row_to_pairwise(row, dataset_cfg), axis=1
    ).tolist()
    
    scoring_config = ScoringConfig(
        n_examples=interpret_cfg.scoring_n_examples,
        max_words_per_example=None,  # Components already truncated in row_to_pairwise_example
        sampling_function=sample_percentile_bins,
        sampling_kwargs=interpret_cfg.scoring_sampling_kwargs,
    )
    
    scoring_metrics = interpreter.score_interpretations(
        texts=train_texts,
        activations=activ_train,
        interpretations=interpretations,
        config=scoring_config,
        prompt_name="pairwise-annotate-singleconcept",
        response_parsing_function=pairwise_response_parsing_function,
    )
    
    neuron_indices = np.arange(activ_train.shape[1])
    best_interpretations = {}
    best_metrics = {}
    for idx in neuron_indices:
        candidates = scoring_metrics.get(int(idx), {})
        if not candidates:
            continue
        best_interp, best = max(
            candidates.items(),
            key=lambda item: item[1].get("correlation", 0.0),
        )
        best_interpretations[int(idx)] = best_interp
        best_metrics[int(idx)] = best
    
    kept_neurons = []
    for idx, metrics in best_metrics.items():
        corr = metrics.get("correlation", 0.0)
        p_value = metrics.get("p_value")
        if corr >= interpret_cfg.min_correlation:
            if interpret_cfg.p_value_threshold is None or (
                p_value is not None and not np.isnan(p_value) and p_value <= interpret_cfg.p_value_threshold
            ):
                kept_neurons.append(idx)
    
    return scoring_metrics, best_interpretations, best_metrics, kept_neurons


def compute_controlled_logit_coefs(
    activ_train: np.ndarray,
    activ_val: np.ndarray,
    train_pred_df: pd.DataFrame,
    val_pred_df: pd.DataFrame,
    train_pred_mask: np.ndarray,
    val_pred_mask: np.ndarray,
    dataset_cfg: DatasetConfig,
    selection_cfg: SelectionSettings,
) -> Dict[int, float]:
    """
    Compute controlled logit coefficients for each SAE feature.
    
    Returns:
        Dictionary mapping neuron_idx to logit coefficient
    """
    activ_train_pred = activ_train[train_pred_mask]
    activ_val_pred = activ_val[val_pred_mask]
    
    control_train = train_pred_df[selection_cfg.controls].to_numpy() if selection_cfg.controls else np.empty((len(train_pred_df), 0))
    control_val = val_pred_df[selection_cfg.controls].to_numpy() if selection_cfg.controls else np.empty((len(val_pred_df), 0))
    
    try:
        all_idx_sorted, all_coefs = select_neurons_controlled_ols(
            activations=activ_train_pred,
            target=train_pred_df[dataset_cfg.label_column].to_numpy(),
            controls=control_train if control_train.size else np.zeros((len(control_train), 0)),
            n_select=activ_train_pred.shape[1],
            classification=selection_cfg.classification,
            standardize=selection_cfg.standardize,
        )
        return {int(idx): float(coef) for idx, coef in zip(all_idx_sorted, all_coefs)}
    except Exception as exc:
        print(f"[selection] controlled OLS failed: {exc}")
        return {}


def run_wimhf_pipeline(config: WIMHFConfig) -> Dict[str, Any]:
    """Run the full WIMHF pipeline and return intermediate artefacts."""
    dataset_cfg = config.dataset
    embedding_cfg = config.embedding
    sae_cfg = config.sae
    interpret_cfg = config.interpretation
    selection_cfg = config.selection
    runtime_cfg = config.runtime

    ckpt_dir_value = runtime_cfg.checkpoint_dir
    if not ckpt_dir_value:
        ckpt_path = get_artifact_subdir(
            "checkpoints",
            f"{dataset_cfg.name}_{embedding_cfg.model}",
        )
    else:
        ckpt_path = Path(ckpt_dir_value).expanduser()
    runtime_cfg.checkpoint_dir = str(ckpt_path)

    wandb_run = None
    wandb_settings = runtime_cfg.wandb
    if wandb_settings and wandb_settings.enable:
        try:
            import wandb  # type: ignore

            wandb_config_payload = dict(wandb_settings.config)
            wandb_config_payload.update(
                {
                    "dataset": dataset_cfg.name,
                    "embedding_model": embedding_cfg.model,
                    "sae_M": sae_cfg.M,
                    "sae_K": sae_cfg.K,
                    "debug": runtime_cfg.debug,
                }
            )
            wandb_run = wandb.init(
                project=wandb_settings.project,
                name=wandb_settings.run_name,
                entity=wandb_settings.entity,
                config=wandb_config_payload,
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[wandb] init failed: {exc}")
            wandb_run = None

    df = _load_dataframe(dataset_cfg.path)

    if runtime_cfg.debug:
        max_rows = min(len(df), 10_000)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=dataset_cfg.random_seed).reset_index(drop=True)
        interpret_cfg.interpreter_model = "gpt-4.1-mini"
        interpret_cfg.annotator_model = "gpt-4.1-mini"
        interpret_cfg.abbreviator_model = "gpt-4.1-mini"
        interpret_cfg.scoring_n_examples = max(1, min(interpret_cfg.scoring_n_examples, 100))
        sae_cfg.batch_size = max(32, min(sae_cfg.batch_size, 128))
        interpret_cfg.n_workers_interpretation = min(interpret_cfg.n_workers_interpretation, 16)
        interpret_cfg.n_workers_annotation = min(interpret_cfg.n_workers_annotation, 64)

    _ensure_length_delta(df, dataset_cfg)

    train_df, val_df = create_connected_component_split(
        df,
        dataset_cfg.split_columns,
        train_frac=dataset_cfg.train_split_size,
        seed=dataset_cfg.split_random_seed,
    )

    train_pred_mask = train_df[dataset_cfg.label_column].isin([0, 1]).to_numpy()
    val_pred_mask = val_df[dataset_cfg.label_column].isin([0, 1]).to_numpy()
    train_pred_df = train_df[train_pred_mask].reset_index(drop=True)
    val_pred_df = val_df[val_pred_mask].reset_index(drop=True)

    dedup_train_df = dedup_pair_rows(train_df, label="train")
    dedup_val_df = dedup_pair_rows(val_df, label="val")

    embedding_cache_dir = runtime_cfg.cache_dir
    if embedding_cache_dir is None:
        from .utils import get_artifact_subdir
        embedding_cache_dir = get_artifact_subdir("emb_cache", f"{dataset_cfg.name}_{embedding_cfg.model}", env_override="EMB_CACHE_DIR")
    response2embedding = _embed_unique_responses(train_df, val_df, dataset_cfg, embedding_cfg, embedding_cache_dir)

    delta_train = _build_deltas(train_df, dataset_cfg, response2embedding)
    delta_val = _build_deltas(val_df, dataset_cfg, response2embedding)
    dedup_delta_train = _build_deltas(dedup_train_df, dataset_cfg, response2embedding)
    dedup_delta_val = _build_deltas(dedup_val_df, dataset_cfg, response2embedding)

    augmented_train = _augment_deltas(dedup_delta_train)

    sae = _train_or_load_sae(augmented_train, dedup_delta_val, sae_cfg, runtime_cfg)
    activ_train = sae.get_activations(delta_train)
    activ_val = sae.get_activations(delta_val)
    activ_dedup = sae.get_activations(dedup_delta_train)

    recon_batches = np.array_split(np.concatenate([delta_train, delta_val], axis=0), 32)
    recon_losses = []
    sae_device = next(sae.parameters()).device
    with torch.no_grad():
        for batch in recon_batches:
            batch_tensor = torch.tensor(batch, dtype=torch.float, device=sae_device)
            recon, info = sae(batch_tensor)
            recon_losses.append(sae.compute_loss(batch_tensor, recon, info).item())
    recon_loss = float(np.mean(recon_losses)) if recon_losses else float("nan")

    dedup_texts = dedup_train_df.apply(lambda row: _row_to_pairwise(row, dataset_cfg), axis=1).tolist()
    train_texts = train_df.apply(lambda row: _row_to_pairwise(row, dataset_cfg), axis=1).tolist()

    sampling_config = SamplingConfig(
        function=sample_top_only,
        n_examples=interpret_cfg.interpret_n_examples,
        max_words_per_example=None,  # Components already truncated in row_to_pairwise_example
        sampling_kwargs=interpret_cfg.interpret_sampling_kwargs,
    )
    completion_kwargs = dict(interpret_cfg.completion_kwargs)
    if not completion_kwargs and "gpt-5" in interpret_cfg.interpreter_model:
        completion_kwargs = {"reasoning_effort": "low"}
    llm_config = LLMConfig(
        max_interpretation_tokens=interpret_cfg.max_interpretation_tokens if interpret_cfg.max_interpretation_tokens is not None else 256,
        completion_kwargs=completion_kwargs,
        timeout=interpret_cfg.timeout if interpret_cfg.timeout is not None else 30.0,
    )
    interpret_config = InterpretConfig(
        sampling=sampling_config,
        llm=llm_config,
        n_candidates=interpret_cfg.n_candidates,
    )
    scoring_config = ScoringConfig(
        n_examples=interpret_cfg.scoring_n_examples,
        max_words_per_example=None,  # Components already truncated in row_to_pairwise_example
        sampling_function=sample_percentile_bins,
        sampling_kwargs=interpret_cfg.scoring_sampling_kwargs,
    )

    interpreter = NeuronInterpreter(
        interpreter_model=interpret_cfg.interpreter_model,
        annotator_model=interpret_cfg.annotator_model,
        cache_dir=config.runtime.cache_dir,
        n_workers_interpretation=interpret_cfg.n_workers_interpretation,
        n_workers_annotation=interpret_cfg.n_workers_annotation,
    )

    interpretations = interpreter.interpret_neurons(
        texts=dedup_texts,
        activations=activ_dedup,
        neuron_indices=list(range(activ_dedup.shape[1])),
        config=interpret_config,
    )

    scoring_metrics = interpreter.score_interpretations(
        texts=train_texts,
        activations=activ_train,
        interpretations=interpretations,
        config=scoring_config,
        prompt_name="pairwise-annotate-singleconcept",
        response_parsing_function=pairwise_response_parsing_function,
    )

    neuron_indices = np.arange(activ_train.shape[1])
    firing_rates = {
        int(idx): float((activ_train[:, idx] != 0).mean()) for idx in neuron_indices
    }

    best_interpretations = {}
    best_metrics = {}
    for idx in neuron_indices:
        candidates = scoring_metrics.get(int(idx), {})
        if not candidates:
            continue
        best_interp, best = max(
            candidates.items(),
            key=lambda item: item[1].get("correlation", 0.0),
        )
        best_interpretations[int(idx)] = best_interp
        best_metrics[int(idx)] = best

    correlations = [abs(m.get("correlation", 0.0)) for m in best_metrics.values()]
    mean_abs_corr = float(np.nanmean(correlations)) if correlations else float("nan")

    kept_neurons = []
    for idx, metrics in best_metrics.items():
        corr = metrics.get("correlation", 0.0)
        p_value = metrics.get("p_value")
        if corr >= interpret_cfg.min_correlation:
            if interpret_cfg.p_value_threshold is None or (
                p_value is not None and not np.isnan(p_value) and p_value <= interpret_cfg.p_value_threshold
            ):
                kept_neurons.append(idx)
    print(
        f"[interpretation] Retained {len(kept_neurons)} / {len(neuron_indices)} neurons after fidelity filtering."
    )

    interpretations_list = [best_interpretations.get(int(idx)) for idx in neuron_indices]
    abbreviations = _abbreviate_interpretations(
        [interp or "" for interp in interpretations_list],
        interpret_cfg.abbreviator_model,
    )
    redundancy, redundancy_neighbors, n_components = measure_interpretation_redundancy(
        [interp for interp in interpretations_list if interp], similarity_threshold=0.7
    )

    feature_rows = []
    for i, idx in enumerate(neuron_indices):
        interp = best_interpretations.get(int(idx))
        metrics = best_metrics.get(int(idx), {})
        feature_rows.append(
            {
                "neuron_idx": int(idx),
                "interpretation": interp,
                "abbreviated_interpretation": abbreviations[i] if i < len(abbreviations) else None,
                "prevalence": firing_rates.get(int(idx)),
                "correlation": metrics.get("correlation"),
                "p_value": metrics.get("p_value"),
                "kept": int(idx) in kept_neurons,
                "semantic_redundancy": float(redundancy[i]) if i < len(redundancy) else np.nan,
                "semantic_neighbor": redundancy_neighbors[i] if i < len(redundancy_neighbors) else None,
            }
        )
    feature_table = pd.DataFrame(feature_rows)

    control_train = train_pred_df[selection_cfg.controls].to_numpy() if selection_cfg.controls else np.empty((len(train_pred_df), 0))
    control_val = val_pred_df[selection_cfg.controls].to_numpy() if selection_cfg.controls else np.empty((len(val_pred_df), 0))
    y_train = train_pred_df[dataset_cfg.label_column].to_numpy()
    y_val = val_pred_df[dataset_cfg.label_column].to_numpy()

    activ_train_pred = activ_train[train_pred_mask]
    activ_val_pred = activ_val[val_pred_mask]

    results = {}
    lasso_coef_maps: Dict[int, Dict[int, float]] = {}
    if selection_cfg.use_lasso and kept_neurons:
        kept_idx = np.array(kept_neurons)
        Z_train_kept = activ_train_pred[:, kept_idx]
        Z_val_kept = activ_val_pred[:, kept_idx]
        for top_k in sorted(set(selection_cfg.lasso_top_k + [min(len(kept_idx), max(selection_cfg.lasso_top_k, default=len(kept_idx)))])):
            try:
                selected_rel, coefs = select_neurons_controlled_lasso(
                    activations=Z_train_kept,
                    target=y_train,
                    controls=control_train if control_train.size else None,
                    n_select=min(top_k, len(kept_idx)),
                    classification=selection_cfg.classification,
                    max_samples=selection_cfg.max_samples_lasso,
                    standardize=selection_cfg.standardize,
                )
            except ValueError as exc:
                print(f"[selection] controlled LASSO failed for top-{top_k}: {exc}")
                continue
            selected_global = kept_idx[selected_rel]
            coef_map = {int(idx): float(coef) for idx, coef in zip(selected_global, coefs)}
            lasso_coef_maps[top_k] = coef_map

            X_train = np.concatenate([activ_train_pred[:, selected_global], control_train], axis=1)
            X_val = np.concatenate([activ_val_pred[:, selected_global], control_val], axis=1)
            train_auc, val_auc = fit_linear_model(
                X_train,
                y_train,
                X_val,
                y_val,
                is_binary=selection_cfg.classification,
                standardize=True,
            )
            results[f"lasso_top_{top_k}_train_auc"] = train_auc
            results[f"lasso_top_{top_k}_val_auc"] = val_auc

    try:
        all_idx_sorted, all_coefs = select_neurons_controlled_ols(
            activations=activ_train_pred,
            target=y_train,
            controls=control_train if control_train.size else np.zeros((len(control_train), 0)),
            n_select=activ_train_pred.shape[1],
            classification=selection_cfg.classification,
            standardize=selection_cfg.standardize,
        )
        logit_coef_map = {int(idx): float(coef) for idx, coef in zip(all_idx_sorted, all_coefs)}
    except Exception as exc:
        print(f"[selection] controlled OLS failed: {exc}")
        logit_coef_map = {}

    feature_table["length_controlled_logit_coef"] = feature_table["neuron_idx"].map(logit_coef_map)

    if kept_neurons and not feature_table.empty:
        retained_table = (
            feature_table[feature_table["neuron_idx"].isin(kept_neurons)]
            .dropna(subset=["correlation", "p_value"])
            .copy()
        )
        if not retained_table.empty:
            retained_table = retained_table.sort_values(
                by="length_controlled_logit_coef",
                key=lambda s: s.abs(),
                ascending=False,
            )
            display_columns = [
                "neuron_idx",
                "abbreviated_interpretation",
                "length_controlled_logit_coef",
                "prevalence",
                "correlation",
                "p_value",
            ]
            printable = retained_table[display_columns].fillna({"length_controlled_logit_coef": 0.0})
            with pd.option_context("display.max_rows", 20, "display.max_columns", None, "display.width", 120):
                print("[features] Retained neuron summary:")
                print(printable.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    X_train_full = np.concatenate([activ_train_pred, control_train], axis=1)
    X_val_full = np.concatenate([activ_val_pred, control_val], axis=1)
    train_auc_full, val_auc_full = fit_linear_model(
        X_train_full,
        y_train,
        X_val_full,
        y_val,
        is_binary=selection_cfg.classification,
        standardize=True,
    )
    results["full_sae_train_auc"] = train_auc_full
    results["full_sae_val_auc"] = val_auc_full

    summary_metrics = {
        "quality/mean_abs_fidelity_correlation": mean_abs_corr,
        "quality/num_semantic_components": float(n_components),
        "quality/num_neurons_total": float(len(neuron_indices)),
        "quality/num_neurons_kept": float(len(kept_neurons)),
        "sae/reconstruction_norm_mse_train_val": recon_loss,
    }

    result_payload = {
        "sae": sae,
        "response_embeddings": response2embedding,
        "activations_train": activ_train,
        "activations_val": activ_val,
        "activations_dedup": activ_dedup,
        "feature_table": feature_table,
        "interpretations": interpretations,
        "scoring_metrics": scoring_metrics,
        "kept_neurons": kept_neurons,
        "lasso_coef_maps": lasso_coef_maps,
        "logit_coef_map": logit_coef_map,
        "auc_metrics": results,
        "summary_metrics": summary_metrics,
        "train_df": train_df,
        "val_df": val_df,
        "train_prediction_df": train_pred_df,
        "val_prediction_df": val_pred_df,
    }

    if wandb_run is not None:
        try:  # pragma: no cover - depends on external service
            log_payload = dict(summary_metrics)
            for metric_name, value in results.items():
                log_payload[f"pred/{metric_name}"] = value
            log_payload["kept_neurons"] = float(len(kept_neurons))
            wandb_run.log(log_payload)
        finally:
            wandb_run.finish()

    return result_payload


# -----------------------------------------------------------------------------
# Utility for configs specified in JSON/YAML
# -----------------------------------------------------------------------------


def load_config(path: str) -> WIMHFConfig:
    with open(path, "r") as f:
        data = json.load(f)
    dataset = DatasetConfig(**data["dataset"])
    embedding_cfg = EmbeddingConfig(**data.get("embedding", {}))
    sae_cfg = SAEConfig(**data.get("sae", {}))
    interpretation_cfg = InterpretationSettings(**data.get("interpretation", {}))
    selection_cfg = SelectionSettings(**data.get("selection", {}))
    runtime_data = data.get("runtime", {})
    runtime_kwargs = {k: v for k, v in runtime_data.items() if k != "wandb"}
    if "cache_dir" in runtime_kwargs and runtime_kwargs["cache_dir"]:
        runtime_kwargs["cache_dir"] = Path(runtime_kwargs["cache_dir"])
    runtime_cfg = RuntimeConfig(**runtime_kwargs)
    if "wandb" in runtime_data and runtime_data["wandb"] is not None:
        runtime_cfg.wandb = WandbSettings(**runtime_data["wandb"])
    return WIMHFConfig(
        dataset=dataset,
        embedding=embedding_cfg,
        sae=sae_cfg,
        interpretation=interpretation_cfg,
        selection=selection_cfg,
        runtime=runtime_cfg,
    )
