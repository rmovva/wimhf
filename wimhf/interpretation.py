"""Neuron interpretation helpers for WIMHF."""

from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from .annotate import ANNOTATION_CACHE_DIR, annotate
from .llm_api import get_completion
from .llm_local import get_local_completions, is_local_model
from .utils import (
    get_artifact_subdir,
    load_prompt,
    swap_responses_if_negative,
    truncate_text,
)

INTERPRETATION_CACHE_DIR = get_artifact_subdir("interpretation_cache", env_override="INTERP_CACHE_DIR")


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def sample_top_zero(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    if random_seed is not None:
        np.random.seed(random_seed)

    neuron_acts = activations[:, neuron_idx]
    n_per_class = max(n_examples // 2, 1)

    top_indices = np.argsort(neuron_acts)[-n_per_class:]
    zero_indices = np.where(neuron_acts == 0)[0]
    if len(zero_indices) >= n_per_class:
        zero_indices = np.random.choice(zero_indices, size=n_per_class, replace=False)

    pos_texts = [texts[i] for i in top_indices]
    neg_texts = [texts[i] for i in zero_indices]

    if max_words_per_example:
        pos_texts = [truncate_text(text, max_words=max_words_per_example) for text in pos_texts]
        neg_texts = [truncate_text(text, max_words=max_words_per_example) for text in neg_texts]

    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": neuron_acts[top_indices].tolist(),
        "negative_activations": neuron_acts[zero_indices].tolist(),
    }


def sample_percentile_bins(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: Optional[int] = None,
    high_percentile: Tuple[float, float] = (90, 100),
    low_percentile: Optional[Tuple[float, float]] = None,
    nonzero_only: bool = True,
    random_seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    if random_seed is not None:
        np.random.seed(random_seed)

    neuron_acts = activations[:, neuron_idx]
    n_per_class = max(n_examples // 2, 1)

    if nonzero_only:
        mask = neuron_acts != 0
        values = neuron_acts[mask]
        indices = np.where(mask)[0]
        if values.size == 0:
            raise ValueError(f"No non-zero activations for neuron {neuron_idx}")
    else:
        values = neuron_acts
        indices = np.arange(len(neuron_acts))

    high_mask = (values >= np.percentile(values, high_percentile[0])) & (
        values <= np.percentile(values, high_percentile[1])
    )
    high_indices = indices[high_mask]
    if len(high_indices) >= n_per_class:
        high_sample = np.random.choice(high_indices, size=n_per_class, replace=False)
    else:
        high_sample = high_indices
        print(f"[interpret] insufficient high-percentile examples for neuron {neuron_idx}; using {len(high_indices)}")

    if low_percentile is not None:
        low_mask = (values >= np.percentile(values, low_percentile[0])) & (
            values <= np.percentile(values, low_percentile[1])
        )
        low_indices = indices[low_mask]
    else:
        low_indices = np.where(neuron_acts == 0)[0] if nonzero_only else indices

    if len(low_indices) >= n_per_class:
        low_sample = np.random.choice(low_indices, size=n_per_class, replace=False)
    else:
        low_sample = low_indices

    pos_texts = [texts[i] for i in high_sample]
    neg_texts = [texts[i] for i in low_sample]

    if max_words_per_example:
        pos_texts = [truncate_text(text, max_words=max_words_per_example) for text in pos_texts]
        neg_texts = [truncate_text(text, max_words=max_words_per_example) for text in neg_texts]

    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": neuron_acts[high_sample].tolist(),
        "negative_activations": neuron_acts[low_sample].tolist(),
    }


def sample_top_only(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    percentile_bin: Optional[Tuple[float, float]] = None,
    n_top_select_from: Optional[int] = None,
    random_seed: Optional[int] = None,
    max_words_per_example: Optional[int] = None,
    swap_negative_response_pairs: bool = False,
    nonzero_only: bool = True,
) -> Dict[str, List[str]]:
    if random_seed is not None:
        np.random.seed(random_seed)

    neuron_acts = activations[:, neuron_idx]
    neuron_mag = np.abs(neuron_acts)

    if swap_negative_response_pairs and np.any(neuron_acts < 0):
        texts = swap_responses_if_negative(texts, neuron_acts)

    if percentile_bin is not None:
        if nonzero_only:
            base_mask = neuron_acts != 0
        else:
            base_mask = np.ones_like(neuron_acts, dtype=bool)
        candidate_vals = neuron_mag[base_mask]
        if candidate_vals.size == 0:
            indices = np.arange(len(neuron_acts))
        else:
            low, high = np.percentile(candidate_vals, percentile_bin[0]), np.percentile(
                candidate_vals, percentile_bin[1]
            )
            mask = base_mask & (neuron_mag >= low) & (neuron_mag <= high)
            indices = np.where(mask)[0]
    elif n_top_select_from is not None:
        indices = np.argsort(-neuron_mag)[:n_top_select_from]
    else:
        indices = np.argsort(-neuron_mag)

    if indices.size > n_examples:
        indices = np.random.choice(indices, size=n_examples, replace=False)

    examples = [texts[i] for i in indices]
    if max_words_per_example:
        examples = [truncate_text(text, max_words=max_words_per_example) for text in examples]

    return {"examples": examples, "activations": neuron_acts[indices].tolist()}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    function: Callable[..., Dict[str, Any]] = sample_top_zero
    n_examples: int = 20
    random_seed: Optional[int] = 0
    max_words_per_example: Optional[int] = 256
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    temperature: Optional[float] = None
    max_interpretation_tokens: int = 256
    timeout: float = 30.0
    completion_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterpretConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    n_candidates: int = 1
    prompt_name: str = "interpret-feature-top-pairs"


@dataclass
class ScoringConfig:
    n_examples: int = 100
    max_words_per_example: Optional[int] = 256
    sampling_function: Callable[..., Dict[str, Any]] = sample_top_zero
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Core interpreter
# -----------------------------------------------------------------------------


class NeuronInterpreter:
    def __init__(
        self,
        interpreter_model: str = "gpt-4.1",
        annotator_model: str = "gpt-4.1-mini",
        cache_dir: Optional[Path] = None,
        n_workers_interpretation: int = 50,
        n_workers_annotation: int = 200,
    ):
        self.interpreter_model = interpreter_model
        self.annotator_model = annotator_model
        self.cache_dir = cache_dir
        self.n_workers_interpretation = n_workers_interpretation
        self.n_workers_annotation = n_workers_annotation

    def _format_prompt(
        self,
        formatted_examples: Dict[str, Any],
        config: InterpretConfig,
    ) -> str:
        prompt_template = load_prompt(config.prompt_name)
        prompt_inputs = dict(
            **formatted_examples,
        )
        return prompt_template.format(**prompt_inputs)

    def _parse_interpretation(self, response: str) -> str:
        """Parse raw LLM response into clean interpretation string."""
        response = response.strip()
        if response.startswith('- '):
            response = response[2:]
        if response.startswith('"-'):
            response = response[2:]
        if response.startswith('" -'):
            response = response[3:]
        return response.strip('"')

    def _generate_interpretation(self, prompt: str, config: InterpretConfig) -> str:
        if is_local_model(self.interpreter_model):
            completion = get_local_completions(
                [prompt],
                model=self.interpreter_model,
                max_tokens=config.llm.max_interpretation_tokens,
                show_progress=False,
            )[0]
        else:
            completion_kwargs = dict(config.llm.completion_kwargs)
            if config.llm.temperature is not None:
                completion_kwargs["temperature"] = config.llm.temperature
            
            # GPT-5-specific handling
            is_gpt5 = "gpt-5" in self.interpreter_model.lower()
            max_tokens = config.llm.max_interpretation_tokens
            timeout = config.llm.timeout
            if is_gpt5:
                # Default reasoning_effort to 'low' if not already in completion_kwargs
                if "reasoning_effort" not in completion_kwargs:
                    completion_kwargs["reasoning_effort"] = "low"
                # Default max_interpretation_tokens to 15000 for GPT-5 if using default (256)
                if max_tokens == 256:
                    max_tokens = 15000
                # Default timeout to 120s for GPT-5 if using default (30.0)
                if timeout == 30.0:
                    timeout = 120.0
            
            completion = get_completion(
                prompt=prompt,
                model=self.interpreter_model,
                max_completion_tokens=max_tokens,
                timeout=timeout,
                **completion_kwargs,
            )
        return self._parse_interpretation(completion)

    def interpret_neurons(
        self,
        texts: List[str],
        activations: np.ndarray,
        neuron_indices: List[int],
        config: InterpretConfig,
    ) -> Dict[int, List[str]]:
        interpretation_tasks = []
        results: Dict[int, List[str]] = {idx: [] for idx in neuron_indices}

        sampling_cfg = config.sampling
        for idx in neuron_indices:
            for candidate in range(config.n_candidates):
                seed = None
                if sampling_cfg.random_seed is not None:
                    seed = sampling_cfg.random_seed + candidate
                interpretation_tasks.append((idx, candidate, seed))

        def _task(args: Tuple[int, int, Optional[int]]) -> Tuple[int, int, str]:
            neuron_idx, candidate_idx, seed = args
            formatted_examples = sampling_cfg.function(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                n_examples=sampling_cfg.n_examples,
                max_words_per_example=sampling_cfg.max_words_per_example,
                random_seed=seed,
                **sampling_cfg.sampling_kwargs,
            )
            prompt = self._format_prompt(formatted_examples, config)
            interpretation = self._generate_interpretation(prompt, config)
            return neuron_idx, candidate_idx, interpretation

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers_interpretation) as executor:
            future_to_task = {executor.submit(_task, task): task for task in interpretation_tasks}
            iterator = tqdm(
                concurrent.futures.as_completed(future_to_task),
                total=len(interpretation_tasks),
                desc=f"Generating {config.n_candidates} interpretation(s) per neuron",
            )
            for future in iterator:
                neuron_idx, candidate_idx, _ = future_to_task[future]
                try:
                    _, _, interpretation = future.result()
                except Exception as exc:
                    print(
                        f"[interpret] failed to generate candidate {candidate_idx} for neuron {neuron_idx}: {exc}"
                    )
                    continue
                interpretation = (interpretation or "").strip()
                if interpretation:
                    results[neuron_idx].append(interpretation)

        return {idx: interps for idx, interps in results.items() if interps}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        annotations: np.ndarray,
        labels: np.ndarray,
        activations: np.ndarray,
    ) -> Dict[str, float]:
        unique_annotations = np.unique(annotations)
        annotations_are_binary = np.all(np.isin(unique_annotations, [0, 1]))

        correlation, p_value = (
            pearsonr(activations, annotations) if len(unique_annotations) > 1 else (0.0, 1.0)
        )

        metrics: Dict[str, float] = {"correlation": float(correlation), "p_value": float(p_value)}

        if annotations_are_binary and (0 in labels and 1 in labels):
            true_pos = float(np.mean(annotations[labels == 1]))
            false_pos = float(np.mean(annotations[labels == 0]))
            precision = 1.0 - false_pos
            if true_pos + precision > 0:
                f1 = 2 * true_pos * precision / (true_pos + precision)
            else:
                f1 = 0.0

            metrics.update(
                {
                    "recall": true_pos,
                    "precision": precision,
                    "f1": f1,
                }
            )
        else:
            metrics.update(
                {
                    "recall": 0.0,
                    "precision": 0.0,
                    "f1": 0.0,
                }
            )
        return metrics

    def score_interpretations(
        self,
        texts: List[str],
        activations: np.ndarray,
        interpretations: Dict[int, List[str]],
        config: Optional[ScoringConfig] = None,
        prompt_name: str = "pairwise-annotate-singleconcept",
        response_parsing_function: Optional[Callable[[str], Any]] = None,
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        config = config or ScoringConfig()

        tasks = []
        scoring_info: Dict[int, Dict[str, Any]] = {}

        for neuron_idx, neuron_interps in interpretations.items():
            formatted_examples = config.sampling_function(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                n_examples=config.n_examples,
                max_words_per_example=config.max_words_per_example,
                random_seed=neuron_idx,
                **config.sampling_kwargs,
            )

            eval_texts = formatted_examples["positive_texts"] + formatted_examples["negative_texts"]
            scoring_info[neuron_idx] = {
                "texts": eval_texts,
                "activations": formatted_examples["positive_activations"]
                + formatted_examples["negative_activations"],
                "labels": np.concatenate(
                    [
                        np.ones(len(formatted_examples["positive_texts"])),
                        np.zeros(len(formatted_examples["negative_texts"])),
                    ]
                ),
            }

            for interp in neuron_interps:
                for text in eval_texts:
                    tasks.append((text, interp))

        progress_desc = (
            f"Scoring neuron interpretation fidelity "
            f"({len(interpretations)} neurons × {config.n_examples} examples)"
        )
        cache_path = (
            (self.cache_dir / "interp-scoring.json") if self.cache_dir else None
        )
        annotations = annotate(
            tasks=tasks,
            cache_path=cache_path,
            n_workers=self.n_workers_annotation,
            show_progress=True,
            model=self.annotator_model,
            prompt_name=prompt_name,
            progress_desc=progress_desc,
            response_parsing_function=response_parsing_function,
        )

        all_metrics: Dict[int, Dict[str, Dict[str, float]]] = {}
        for neuron_idx, neuron_interps in interpretations.items():
            neuron_metrics: Dict[str, Dict[str, float]] = {}
            info = scoring_info[neuron_idx]
            texts_for_scoring = info["texts"]
            labels = info["labels"]
            acts = np.array(info["activations"])

            for interp in neuron_interps:
                concept_annots = annotations.get(interp, {})
                mask = np.array([text in concept_annots for text in texts_for_scoring])
                if mask.sum() == 0:
                    neuron_metrics[interp] = {
                        "correlation": 0.0,
                        "p_value": 1.0,
                        "recall": 0.0,
                        "precision": 0.0,
                        "f1": 0.0,
                    }
                    continue

                annot_values = np.array(
                    [concept_annots[text] for text, keep in zip(texts_for_scoring, mask) if keep],
                    dtype=float,
                )
                label_values = labels[mask]
                activation_values = acts[mask]
                neuron_metrics[interp] = self._compute_metrics(annot_values, label_values, activation_values)

            all_metrics[neuron_idx] = neuron_metrics

        return all_metrics


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def save_interpretations(
    filename: str,
    interpretations: Dict[int, List[str]],
    selected_neurons: Optional[List[int]] = None,
    selection_scores: Optional[List[float]] = None,
    selection_method: Optional[str] = None,
) -> None:
    payload = {"interpretations": interpretations}
    if selected_neurons is not None:
        payload["selected_neurons"] = selected_neurons
    if selection_scores is not None:
        payload["selection_scores"] = selection_scores
    if selection_method is not None:
        payload["selection_method"] = selection_method

    path = INTERPRETATION_CACHE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
