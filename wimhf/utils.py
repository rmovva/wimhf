"""Shared utilities for the WIMHF pipeline."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import tiktoken
from scipy.special import expit  # noqa: F401  # imported for backwards compatibility
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TOKENIZER = tiktoken.get_encoding("cl100k_base")
_PROMPT_CACHE: Dict[str, str] = {}
_PROMPT_ALIASES: Dict[str, str] = {
    "annotate-pairwise": "pairwise-annotate-singleconcept",
    "annotate-multiconcept-pairwise": "pairwise-annotate-multiconcept",
    "interpret-neuron-pairwise-v3": "interpret-feature-top-pairs",
}


def get_repo_root() -> Path:
    """Return the repository root (parent of the wimhf package)."""
    return Path(__file__).resolve().parent.parent


def get_artifact_root() -> Path:
    """Return the base directory for large artefacts (cache, checkpoints, etc.)."""
    base = os.environ.get("WIMHF_ARTIFACTS_DIR")
    return Path(base).expanduser() if base else get_repo_root()


def get_artifact_subdir(
    *parts: str,
    env_override: Optional[str] = None,
    create: bool = False,
) -> Path:
    """
    Resolve a subdirectory under the artefact root, optionally overridden by a specific
    environment variable. Set ``create=True`` to ensure the directory exists.
    """
    if env_override:
        override = os.environ.get(env_override)
        if override:
            path = Path(override).expanduser()
            if create:
                path.mkdir(parents=True, exist_ok=True)
            return path

    path = get_artifact_root().joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def load_prompt(prompt_name: str) -> str:
    """Load a prompt template from ``wimhf/prompts`` with caching."""
    canonical_name = _PROMPT_ALIASES.get(prompt_name, prompt_name)

    if prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]
    if canonical_name in _PROMPT_CACHE:
        _PROMPT_CACHE[prompt_name] = _PROMPT_CACHE[canonical_name]
        return _PROMPT_CACHE[canonical_name]

    prompt_path = Path(__file__).parent / "prompts" / f"{canonical_name}.txt"
    try:
        content = prompt_path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}. Ensure the template exists under wimhf/prompts."
        ) from exc

    _PROMPT_CACHE[canonical_name] = content
    _PROMPT_CACHE[prompt_name] = content
    return content


def truncate_text(
    text: str,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    truncation_message: str = "[...rest of text is truncated]",
    left_truncation_message: str = "[start of the conversation is truncated, only showing the end...]",
    left_truncate: bool = False,
) -> str:
    """
    Truncate text based on words, characters, or tokens while preserving whitespace when possible.
    """
    if all(x is None for x in (max_words, max_chars, max_tokens)):
        return text

    if left_truncate and text.startswith(left_truncation_message):
        return text
    if not left_truncate and text.endswith(truncation_message):
        return text

    truncated = text

    if max_words is not None:
        matches = list(re.finditer(r"\S+", text))
        total_words = len(matches)
        if total_words > max_words:
            if left_truncate:
                start_idx = matches[-max_words].start()
                truncated = text[start_idx:]
            else:
                end_idx = matches[max_words - 1].end()
                truncated = text[:end_idx]

    if max_chars is not None and len(truncated) > max_chars:
        truncated = truncated[-max_chars:] if left_truncate else truncated[:max_chars]

    if max_tokens is not None:
        tokens = TOKENIZER.encode(truncated)
        if len(tokens) > max_tokens:
            truncated = TOKENIZER.decode(tokens[-max_tokens:] if left_truncate else tokens[:max_tokens])

    if truncated != text:
        truncated = (
            f"{left_truncation_message}{truncated}" if left_truncate else f"{truncated}{truncation_message}"
        )
    return truncated


def get_text_for_printing(text: str, max_chars: int = 128) -> str:
    """Truncate and remove newlines from a string for compact printing."""
    return truncate_text(text, max_chars=max_chars).replace("\n", " ")


def add_line_breaks(text: str, chars_per_line: int = 100, prepend_tabs: int = 0) -> str:
    """Insert line breaks every ``chars_per_line`` characters without breaking words."""
    tab_str = "\t" * max(prepend_tabs, 0)
    words = text.split()
    lines: List[str] = []
    current_line: List[str] = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1
        if current_line and current_length + word_len > chars_per_line:
            lines.append(tab_str + " ".join(current_line))
            current_line = [word]
            current_length = word_len
        else:
            current_line.append(word)
            current_length += word_len

    if current_line:
        lines.append(tab_str + " ".join(current_line))
    return "\n".join(lines)


def print_wrapped(text: str, line_width: int = 60, prepend_tabs: int = 0) -> str:
    """
    Wrap text for readable printing by adding line breaks.
    
    Args:
        text: Text to wrap
        line_width: Maximum characters per line before wrapping (default: 60)
        prepend_tabs: Number of tabs to prepend to each line (default: 0)
        
    Returns:
        Text with line breaks added for readability
    """
    # Replace newlines with spaces first to normalize
    normalized = text.replace("\n", " ")
    # Add line breaks
    return add_line_breaks(normalized, chars_per_line=line_width, prepend_tabs=prepend_tabs)


def filter_invalid_texts(texts: Iterable[str]) -> List[str]:
    """Filter out ``None`` values and empty strings from a collection of texts."""
    return [t for t in texts if t is not None and str(t).strip()]


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON, creating directories as needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON if the file exists, otherwise return an empty dict."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


def fit_linear_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_reg: float = 1.0,
    is_binary: Optional[bool] = None,
    standardize: bool = True,
) -> Tuple[float, float]:
    """
    Fit a simple linear model (logistic regression or ridge) and return train/val scores.
    Classification uses ROC-AUC; regression uses Pearson correlation.
    """
    if standardize:
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_val_std = scaler.transform(X_val)
    else:
        X_train_std = X_train
        X_val_std = X_val

    if is_binary is None:
        is_binary = np.unique(y_train).tolist() in ([0, 1], [0], [1])

    if is_binary:
        model = LogisticRegression(C=C_reg, max_iter=1000)
        model.fit(X_train_std, y_train)
        train_score = roc_auc_score(y_train, model.predict_proba(X_train_std)[:, 1])
        val_score = roc_auc_score(y_val, model.predict_proba(X_val_std)[:, 1])
    else:
        model = Ridge(alpha=C_reg)
        model.fit(X_train_std, y_train)
        train_score = pearsonr(y_train, model.predict(X_train_std))[0]
        val_score = pearsonr(y_val, model.predict(X_val_std))[0]

    return float(train_score), float(val_score)


def swap_responses(pairwise_text: str) -> str:
    """Swap RESPONSE A/B blocks inside a formatted pairwise prompt string."""
    return re.sub(
        r'(RESPONSE A: "Assistant: )(.*?)("\n\nRESPONSE B: "Assistant: )(.*?)(")',
        r"\1\4\3\2\5",
        pairwise_text,
        flags=re.S,
    )


def swap_responses_if_negative(pairwise_texts: Iterable[str], activations: np.ndarray) -> List[str]:
    """Swap A/B responses whenever the activation is negative."""
    return [
        swap_responses(text) if activation < 0 else text
        for text, activation in zip(pairwise_texts, activations)
    ]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def row_to_prompt_response(row: pd.Series, response_col: str) -> str:
    """Create ``prompt + response`` text for reward-model training."""
    prompt = row["prompt"]
    if prompt.startswith("Human: "):
        prompt_text = prompt
    else:
        prompt_text = f"Human: {prompt}"
    response = row[response_col]
    return f"{prompt_text}\n\nAssistant: {response}"


def row_to_pairwise_example(
    row: pd.Series,
    include_prompt: bool = True,
    max_words_prompt: int = 256,
    max_words_response: int = 256,
) -> str:
    """Format a dataframe row into a pairwise comparison prompt."""
    prompt = row["prompt"]
    if prompt.endswith("\n\nAssistant:"):
        prompt = prompt[:-13]
    if not prompt.startswith("Human: "):
        prompt = f"Human: {prompt}"

    prompt = truncate_text(prompt, max_words=max_words_prompt, left_truncate=True)
    response_a = truncate_text(row["response_A"], max_words=max_words_response)
    response_b = truncate_text(row["response_B"], max_words=max_words_response)

    if include_prompt:
        return (
            f'CONTEXT: "{prompt}"\n\n'
            f'RESPONSE A: "Assistant: {response_a}"\n\n'
            f'RESPONSE B: "Assistant: {response_b}"'
        )
    return (
        f'RESPONSE A: "Assistant: {response_a}"\n\n'
        f'RESPONSE B: "Assistant: {response_b}"'
    )


def pairwise_response_parsing_function(response_text: str) -> Optional[int]:
    """Parse LLM outputs for pairwise annotations into {-1, 0, 1}."""
    if response_text == "a":
        return 1
    if response_text == "b":
        return -1
    if response_text == "tie":
        return 0
    return None


def dedup_pair_rows(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Deduplicate (response_A, response_B) pairs treating (A,B) and (B,A) as the same.
    """
    df = df.copy()
    pair_keys = [tuple(sorted((a, b))) for a, b in zip(df["response_A"], df["response_B"])]
    df["__pair_key__"] = pair_keys

    dedup = df.groupby("__pair_key__").sample(n=1, random_state=42).reset_index(drop=True)
    if label:
        print(f"[dedup {label}] {len(df)} -> {len(dedup)} rows (unique unordered pairs)")
    return dedup.drop(columns=["__pair_key__"])


def create_connected_component_split(
    df: pd.DataFrame,
    split_columns: List[str],
    train_frac: float = 0.7,
    seed: int = 42,
    return_masks: bool = False,
    verbose: bool = True,
):
    """
    Split a dataframe into train/validation sets while keeping all rows sharing the same
    ``split_columns`` values in the same split (to avoid leakage).
    """
    np.random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))

    if verbose:
        print("Building graph edges...")

    for col in split_columns:
        value_to_rows: Dict[Any, List[int]] = {}
        for idx, val in enumerate(df[col]):
            value_to_rows.setdefault(val, []).append(idx)

        for rows in value_to_rows.values():
            if len(rows) <= 1:
                continue
            if len(rows) > 1000:
                edges = [(rows[0], rows[i]) for i in range(1, len(rows))]
            else:
                edges = [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
            G.add_edges_from(edges)

    if verbose:
        print("\nFinding connected components...")
    components = list(nx.connected_components(G))
    np.random.shuffle(components)

    train_indices: List[int] = []
    total_rows = 0
    target_train = len(df) * train_frac

    for component in components:
        if total_rows < target_train:
            train_indices.extend(component)
            total_rows += len(component)

    mask = np.zeros(len(df), dtype=bool)
    mask[train_indices] = True
    train_df = df[mask].reset_index(drop=True)
    val_df = df[~mask].reset_index(drop=True)

    if verbose:
        print(
            f"Connected-component split -> train {len(train_df)} ({len(train_df)/len(df):.1%}), "
            f"val {len(val_df)} ({len(val_df)/len(df):.1%})"
        )

    if return_masks:
        return train_df, val_df, mask, ~mask
    return train_df, val_df


def measure_interpretation_redundancy(
    neuron_interpretations: List[str],
    similarity_threshold: float = 0.75,
) -> Tuple[np.ndarray, List[str], int]:
    """Compute redundancy metrics by embedding interpretations and measuring cosine similarity."""
    from .embedding import get_openai_embeddings  # local import to avoid circular dependency

    embeddings_map = get_openai_embeddings(neuron_interpretations, show_progress=False)
    embeddings = np.array([embeddings_map[text] for text in neuron_interpretations])
    n = len(neuron_interpretations)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            denom = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            if denom == 0:
                continue
            similarity_matrix[i, j] = float(np.dot(embeddings[i], embeddings[j]) / denom)

    max_similarities = similarity_matrix.max(axis=1)
    nearest_neighbors = [
        neuron_interpretations[int(np.argmax(similarity_matrix[i]))] for i in range(n)
    ]

    adjacency_matrix = similarity_matrix > similarity_threshold
    np.fill_diagonal(adjacency_matrix, False)
    components = list(nx.connected_components(nx.from_numpy_array(adjacency_matrix)))
    num_components = len(components)

    return max_similarities, nearest_neighbors, num_components
