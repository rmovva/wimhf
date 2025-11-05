"""Baseline reward modelling helpers for WIMHF."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft import LoraConfig, PeftModel, TaskType

from trl import RewardConfig, RewardTrainer

from .embedding import get_openai_embeddings
from .feature_selection import select_neurons_controlled_lasso
from .reward_utils import load_chat_tokenizer, prepare_reward_dataset
from .sae import load_model as load_sae_model
from .utils import create_connected_component_split, get_artifact_subdir, row_to_prompt_response



@dataclass
class RewardDatasetConfig:
    name: str
    path: str
    split_columns: List[str]
    train_split_size: float = 0.8
    split_random_seed: int = 42
    label_column: str = "label"
    prompt_column: str = "prompt"
    response_a_column: str = "response_A"
    response_b_column: str = "response_B"
    sae_checkpoint_path: Optional[str] = None


@dataclass
class EmbeddingBaselineConfig:
    embedder: str = "text-embedding-3-small"
    n_workers: int = 2
    cache_dir: Optional[Path] = None


@dataclass
class RewardModelConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    learning_rate: float = 5e-5
    lora_rank: int = 16
    num_epochs: int = 1
    train_batch_size: int = 4
    eval_batch_size: int = 16
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    gradient_accumulation: Optional[int] = 4
    num_workers: Optional[int] = None
    disable_wandb: bool = True
    reuse_checkpoint: Optional[str] = None


@dataclass
class RewardBaselineResult:
    dataset_name: str
    auc_embed_prompt_response: float
    acc_embed_prompt_response: float
    auc_embed_response_only: float
    acc_embed_response_only: float
    auc_sae_full: float
    acc_sae_full: float
    auc_sae_top5: float
    acc_sae_top5: float
    auc_sae_top10: float
    acc_sae_top10: float
    auc_reward_model: Optional[float] = None
    acc_reward_model: Optional[float] = None


def _read_df(path: str) -> pd.DataFrame:
    if path.endswith(".json") or path.endswith(".jsonl"):
        return pd.read_json(path, orient="records", lines=True)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset extension: {path}")


def _build_delta_matrix(
    df: pd.DataFrame,
    cfg: RewardDatasetConfig,
    text2emb: Dict[str, np.ndarray],
    make_text_fn,
) -> np.ndarray:
    ea = np.stack([text2emb[make_text_fn(row, cfg.response_a_column)] for _, row in df.iterrows()])
    eb = np.stack([text2emb[make_text_fn(row, cfg.response_b_column)] for _, row in df.iterrows()])
    return ea - eb


def _make_text_prompt_plus_response(row: pd.Series, which: str) -> str:
    return row_to_prompt_response(row, which)


def _fit_eval_logreg(
    E_tr: np.ndarray,
    y_tr: np.ndarray,
    E_va: np.ndarray,
    y_va: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(E_tr, y_tr)
    probs = clf.predict_proba(E_va)[:, 1]
    auc = float(roc_auc_score(y_va, probs))
    acc = float(accuracy_score(y_va, (probs >= 0.5).astype(int)))
    return auc, acc, probs


def run_embedding_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: RewardDatasetConfig,
    emb_cfg: EmbeddingBaselineConfig,
    text2emb: Dict[str, np.ndarray],
    controls_train: np.ndarray,
    controls_val: np.ndarray,
    make_text_fn=_make_text_prompt_plus_response,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    E_tr = _build_delta_matrix(train_df, cfg, text2emb, make_text_fn)
    E_va = _build_delta_matrix(val_df, cfg, text2emb, make_text_fn)
    y_tr = train_df[cfg.label_column].to_numpy()
    y_va = val_df[cfg.label_column].to_numpy()

    X_tr = np.concatenate([E_tr, controls_train], axis=1)
    X_va = np.concatenate([E_va, controls_val], axis=1)

    auc, acc, _ = _fit_eval_logreg(X_tr, y_tr, X_va, y_va)
    return auc, acc, text2emb


def _prepare_reward_dataframe(df: pd.DataFrame, cfg: RewardDatasetConfig) -> pd.DataFrame:
    chosen = np.where(df[cfg.label_column].to_numpy() == 1, df[cfg.response_a_column], df[cfg.response_b_column])
    rejected = np.where(df[cfg.label_column].to_numpy() == 1, df[cfg.response_b_column], df[cfg.response_a_column])
    return pd.DataFrame(
        {
            "prompt": df[cfg.prompt_column].values,
            "chosen": chosen,
            "rejected": rejected,
        }
    )


def train_reward_model(
    train_df: pd.DataFrame,
    cfg: RewardDatasetConfig,
    rm_cfg: RewardModelConfig,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    grad_acc = rm_cfg.gradient_accumulation or 4
    if rm_cfg.num_workers is None:
        cpu_ct = os.cpu_count() or 8
        rm_cfg.num_workers = min(8, max(1, cpu_ct - 1))
    num_workers = rm_cfg.num_workers

    tokenizer = load_chat_tokenizer(rm_cfg.model_name)
    ds = prepare_reward_dataset(_prepare_reward_dataframe(train_df, cfg), tokenizer, rm_cfg.model_name)

    if rm_cfg.reuse_checkpoint:
        save_dir_path = Path(rm_cfg.reuse_checkpoint).expanduser()
        save_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        save_dir_path = get_artifact_subdir(
            "reward_models",
            f"RM_{cfg.name}_{rm_cfg.model_name.split('/')[-1]}",
            create=True,
        )
    save_dir = str(save_dir_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        rm_cfg.model_name,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        if getattr(model.generation_config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=rm_cfg.lora_rank,
        lora_alpha=rm_cfg.lora_rank * 2,
        lora_dropout=0.1,
        inference_mode=False,
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
        bias="lora_only",
    )

    optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    trainer = RewardTrainer(
        model=model,
        train_dataset=ds,
        args=RewardConfig(
            output_dir=save_dir,
            num_train_epochs=rm_cfg.num_epochs,
            per_device_train_batch_size=rm_cfg.train_batch_size,
            per_device_eval_batch_size=rm_cfg.eval_batch_size,
            learning_rate=rm_cfg.learning_rate,
            gradient_accumulation_steps=grad_acc,
            warmup_ratio=rm_cfg.warmup_ratio,
            lr_scheduler_type=rm_cfg.lr_scheduler_type,
            logging_steps=50,
            eval_steps=0,
            eval_strategy="no",
            save_strategy="no",
            seed=42,
            remove_unused_columns=False,
            max_length=rm_cfg.max_seq_length,
            dataloader_drop_last=True,
            optim=optim,
            dataloader_num_workers=num_workers,
            bf16=torch.cuda.is_available(),
            group_by_length=True,
            length_column_name="length",
            report_to=[] if rm_cfg.disable_wandb else ["wandb"],
        ),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model()

    trained_model = trainer.model
    if isinstance(trained_model, PeftModel):
        trained_model = trained_model.merge_and_unload()

    trained_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    if torch.cuda.is_available():
        trained_model = trained_model.to("cuda")
    trained_model.eval()

    return trained_model, tokenizer


def _score_reward_side(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    df_pairs: pd.DataFrame,
    cfg: RewardDatasetConfig,
    side: str,
    rm_cfg: RewardModelConfig,
) -> np.ndarray:
    df_side = pd.DataFrame(
        {
            "prompt": df_pairs[cfg.prompt_column].values,
            "chosen": df_pairs[side].values,
            "rejected": df_pairs[side].values,
        }
    )
    dataset = prepare_reward_dataset(df_side, tokenizer, rm_cfg.model_name)
    texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in dataset["chosen"]]  # type: ignore[index]

    device = next(model.parameters()).device
    scores: List[torch.Tensor] = []
    batch_size = rm_cfg.eval_batch_size

    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=rm_cfg.max_seq_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits.squeeze(-1).float()
        scores.append(logits.cpu())

    return torch.cat(scores).numpy()


def evaluate_reward_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    val_df: pd.DataFrame,
    cfg: RewardDatasetConfig,
    rm_cfg: RewardModelConfig,
) -> Tuple[float, float]:
    score_a = _score_reward_side(model, tokenizer, val_df, cfg, cfg.response_a_column, rm_cfg)
    score_b = _score_reward_side(model, tokenizer, val_df, cfg, cfg.response_b_column, rm_cfg)
    preds = score_a - score_b
    labels = val_df[cfg.label_column].to_numpy()
    auc = float(roc_auc_score(labels, preds))
    acc = float(accuracy_score(labels, (preds > 0).astype(int)))
    return auc, acc


def _dedup_train_pairs(df: pd.DataFrame, cfg: RewardDatasetConfig) -> Tuple[pd.DataFrame, int]:
    group_cols = [cfg.prompt_column, cfg.response_a_column, cfg.response_b_column]
    grp = df.groupby(group_cols, as_index=False)
    agg = grp.agg(
        {
            cfg.label_column: lambda x: int(np.round(np.mean(x))),
            "length_delta": "first",
        }
    )
    columns = [
        cfg.prompt_column,
        cfg.response_a_column,
        cfg.response_b_column,
        cfg.label_column,
        "length_delta",
    ]
    dropped = int(len(df) - len(agg))
    return agg[columns].reset_index(drop=True), dropped


def _collect_embedding_texts(
    dfs: List[pd.DataFrame],
    cfg: RewardDatasetConfig,
    make_text_fn,
) -> List[str]:
    texts: List[str] = []
    for df in dfs:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            texts.append(make_text_fn(row, cfg.response_a_column))
            texts.append(make_text_fn(row, cfg.response_b_column))
    return list(dict.fromkeys(texts))


def run_reward_baselines(
    cfg: RewardDatasetConfig,
    emb_cfg: EmbeddingBaselineConfig,
    rm_cfg: Optional[RewardModelConfig] = None,
) -> RewardBaselineResult:
    df = _read_df(cfg.path)
    if "length_delta" not in df.columns:
        df["length_delta"] = df[cfg.response_a_column].str.split().str.len() - df[cfg.response_b_column].str.split().str.len()

    train_df, val_df = create_connected_component_split(
        df,
        cfg.split_columns,
        train_frac=cfg.train_split_size,
        seed=cfg.split_random_seed,
    )

    train_df_dedup_all, _ = _dedup_train_pairs(train_df, cfg)

    val_df_binary = val_df[val_df[cfg.label_column].isin([0, 1])].reset_index(drop=True)
    if val_df_binary.empty:
        raise ValueError(f"{cfg.name}: no binary-label rows available in validation split.")

    train_binary_raw = train_df[train_df[cfg.label_column].isin([0, 1])].reset_index(drop=True)
    train_df_binary, _ = _dedup_train_pairs(train_binary_raw, cfg)
    if train_df_binary.empty:
        raise ValueError(f"{cfg.name}: no binary-label rows available in training split after deduplication.")

    scaler_ctrl = StandardScaler()
    control_train = scaler_ctrl.fit_transform(train_df_binary[["length_delta"]].to_numpy())
    control_val = scaler_ctrl.transform(val_df_binary[["length_delta"]].to_numpy())

    make_pr = _make_text_prompt_plus_response
    make_resp = lambda row, which: row[which]

    corpus_pr = _collect_embedding_texts([train_df_dedup_all, val_df], cfg, make_pr)
    cache_dir_pr = emb_cfg.cache_dir
    if cache_dir_pr is None:
        from .utils import get_artifact_subdir
        cache_dir_pr = get_artifact_subdir("emb_cache", f"{cfg.name}_{emb_cfg.embedder}_prompt_plus_response", env_override="EMB_CACHE_DIR")
    text2emb_pr = get_openai_embeddings(
        texts=corpus_pr,
        model=emb_cfg.embedder,
        n_workers=emb_cfg.n_workers,
        cache_dir=cache_dir_pr,
    )

    auc_embed_pr, acc_embed_pr, _ = run_embedding_baseline(
        train_df_binary, val_df_binary, cfg, emb_cfg, text2emb_pr, control_train, control_val
    )

    corpus_resp = _collect_embedding_texts([train_df_binary, val_df_binary], cfg, make_resp)
    cache_dir_resp = get_artifact_subdir("emb_cache", f"{cfg.name}_{emb_cfg.embedder}_response_only", env_override="EMB_CACHE_DIR")
    text2emb_resp = get_openai_embeddings(
        texts=corpus_resp,
        model=emb_cfg.embedder,
        n_workers=emb_cfg.n_workers,
        cache_dir=cache_dir_resp,
    )

    auc_embed_resp, acc_embed_resp, _ = run_embedding_baseline(
        train_df_binary,
        val_df_binary,
        cfg,
        emb_cfg,
        text2emb_resp,
        control_train,
        control_val,
        make_text_fn=make_resp,
    )

    auc_sae_full = acc_sae_full = None
    auc_sae_top5 = acc_sae_top5 = None
    auc_sae_top10 = acc_sae_top10 = None
    if cfg.sae_checkpoint_path and os.path.exists(cfg.sae_checkpoint_path):
        sae_model = load_sae_model(cfg.sae_checkpoint_path)
        delta_train_all = _build_delta_matrix(train_df_dedup_all, cfg, text2emb_pr, make_pr)
        delta_val = _build_delta_matrix(val_df_binary, cfg, text2emb_pr, make_pr)
        Z_tr_all = sae_model.get_activations(delta_train_all)
        Z_va = sae_model.get_activations(delta_val)

        mask_binary = train_df_dedup_all[cfg.label_column].isin([0, 1]).to_numpy()
        Z_tr = Z_tr_all[mask_binary]
        y_tr = train_df_binary[cfg.label_column].to_numpy()
        y_va = val_df_binary[cfg.label_column].to_numpy()
        controls_tr = control_train
        controls_va = control_val

        scaler_full = StandardScaler()
        X_tr_full = scaler_full.fit_transform(np.concatenate([Z_tr, controls_tr], axis=1))
        X_va_full = scaler_full.transform(np.concatenate([Z_va, controls_va], axis=1))
        clf_full = LogisticRegression(max_iter=1000)
        clf_full.fit(X_tr_full, y_tr)
        probs_full = clf_full.predict_proba(X_va_full)[:, 1]
        auc_sae_full = float(roc_auc_score(y_va, probs_full))
        acc_sae_full = float(accuracy_score(y_va, (probs_full >= 0.5).astype(int)))

        top_k_values = [5, 10]
        max_k = min(max(top_k_values), Z_tr.shape[1])
        if max_k > 0:
            selected_rel, _ = select_neurons_controlled_lasso(
                activations=Z_tr,
                target=y_tr,
                controls=controls_tr,
                n_select=max_k,
                classification=True,
                standardize=True,
            )
            selected_rel = list(selected_rel)
            for k in top_k_values:
                if k > len(selected_rel):
                    continue
                subset = selected_rel[:k]
                scaler_k = StandardScaler()
                X_tr_k = scaler_k.fit_transform(np.concatenate([Z_tr[:, subset], controls_tr], axis=1))
                X_va_k = scaler_k.transform(np.concatenate([Z_va[:, subset], controls_va], axis=1))
                clf_k = LogisticRegression(max_iter=1000)
                clf_k.fit(X_tr_k, y_tr)
                probs_k = clf_k.predict_proba(X_va_k)[:, 1]
                auc_k = float(roc_auc_score(y_va, probs_k))
                acc_k = float(accuracy_score(y_va, (probs_k >= 0.5).astype(int)))
                if k == 5:
                    auc_sae_top5, acc_sae_top5 = auc_k, acc_k
                if k == 10:
                    auc_sae_top10, acc_sae_top10 = auc_k, acc_k

    auc_rm = acc_rm = None
    if rm_cfg is not None:
        rm_model, rm_tokenizer = train_reward_model(train_df_binary, cfg, rm_cfg)
        auc_rm, acc_rm = evaluate_reward_model(rm_model, rm_tokenizer, val_df_binary, cfg, rm_cfg)

    return RewardBaselineResult(
        dataset_name=cfg.name,
        auc_embed_prompt_response=auc_embed_pr,
        acc_embed_prompt_response=acc_embed_pr,
        auc_embed_response_only=auc_embed_resp,
        acc_embed_response_only=acc_embed_resp,
        auc_sae_full=auc_sae_full if auc_sae_full is not None else float("nan"),
        acc_sae_full=acc_sae_full if acc_sae_full is not None else float("nan"),
        auc_sae_top5=auc_sae_top5 if auc_sae_top5 is not None else float("nan"),
        acc_sae_top5=acc_sae_top5 if acc_sae_top5 is not None else float("nan"),
        auc_sae_top10=auc_sae_top10 if auc_sae_top10 is not None else float("nan"),
        acc_sae_top10=acc_sae_top10 if acc_sae_top10 is not None else float("nan"),
        auc_reward_model=auc_rm,
        acc_reward_model=acc_rm,
    )
