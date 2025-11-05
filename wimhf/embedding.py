"""Embedding utilities for WIMHF."""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tiktoken
import torch
from tqdm.auto import tqdm

from .llm_api import get_client
from .utils import filter_invalid_texts

enc = tiktoken.get_encoding("cl100k_base")


def _embed_batch_openai(
    batch: List[str],
    model: str,
    client,
    max_tokens: int = 8192,
    max_retries: int = 3,
    backoff_factor: float = 3.0,
    timeout: float = 10.0,
) -> List[List[float]]:
    truncated_batch = []
    for text in batch:
        tokens = enc.encode(text.strip(), disallowed_special=())
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        truncated_batch.append(text)

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=truncated_batch,
                model=model,
                timeout=timeout,
            )
            return [record.embedding for record in response.data]
        except Exception as exc:
            if attempt == max_retries - 1:
                raise exc
            wait_time = timeout * (backoff_factor**attempt)
            print(f"[embedding] retrying after {wait_time:.1f}s ({attempt + 1}/{max_retries}) due to {exc}")
            time.sleep(wait_time)
    raise RuntimeError("Unreachable")


def load_embedding_cache(cache_dir: Optional[Path]) -> Dict[str, np.ndarray]:
    if not cache_dir:
        return {}

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {}

    text2embedding: Dict[str, np.ndarray] = {}
    chunk_files = sorted(glob.glob(str(cache_path / "chunk_*.npy")))
    for chunk in tqdm(chunk_files, desc="Loading embedding cache"):
        data = np.load(chunk, allow_pickle=True)
        for text, emb in data:
            text2embedding[text] = emb
    return text2embedding


def update_embedding_cache(
    cache_dir: Optional[Path],
    text2embedding: Dict[str, np.ndarray],
    chunk_size: int = 50_000,
) -> None:
    if not cache_dir or not text2embedding:
        return

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    items = list(text2embedding.items())
    for idx in range(0, len(items), chunk_size):
        chunk_items = items[idx : idx + chunk_size]
        chunk_path = cache_path / f"chunk_{idx // chunk_size:03d}.npy"
        np.save(chunk_path, np.array(chunk_items, dtype=object))


def _get_next_chunk_index(cache_dir: Optional[Path]) -> int:
    if not cache_dir:
        return 0
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return 0
    chunk_files = glob.glob(str(cache_path / "chunk_*.npy"))
    if not chunk_files:
        return 0
    indices = [int(Path(path).stem.split("_")[1]) for path in chunk_files]
    return max(indices) + 1


def _save_embedding_chunk(
    cache_dir: Optional[Path],
    chunk_embeddings: Dict[str, np.ndarray],
    chunk_idx: int,
) -> int:
    if not cache_dir or not chunk_embeddings:
        return chunk_idx
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    chunk_path = cache_path / f"chunk_{chunk_idx:03d}.npy"
    np.save(chunk_path, np.array(list(chunk_embeddings.items()), dtype=object))
    print(f"[embedding] saved {len(chunk_embeddings)} items to {chunk_path}")
    return chunk_idx + 1


def get_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
    n_workers: int = 5,
    cache_dir: Optional[Path] = None,
    show_progress: bool = True,
    chunk_size: int = 50_000,
    timeout: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Embed texts using the OpenAI API with chunked caching."""
    texts = filter_invalid_texts(texts)
    cache = load_embedding_cache(cache_dir)

    to_embed = [text for text in texts if text not in cache]
    if not to_embed:
        return cache

    client = get_client()
    next_chunk_idx = _get_next_chunk_index(cache_dir)

    for start in tqdm(range(0, len(to_embed), chunk_size), desc="Embedding chunks", disable=not show_progress):
        chunk_texts = to_embed[start : start + chunk_size]
        chunk_embeddings: Dict[str, np.ndarray] = {}

        for batch_start in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[batch_start : batch_start + batch_size]
            embeddings = _embed_batch_openai(batch, model, client, timeout=timeout)
            for text, emb in zip(batch, embeddings):
                cache[text] = np.array(emb, dtype=np.float32)
                chunk_embeddings[text] = cache[text]

        next_chunk_idx = _save_embedding_chunk(cache_dir, chunk_embeddings, next_chunk_idx)

    return cache


def get_local_embeddings(
    texts: List[str],
    model: str,
    batch_size: int = 64,
    show_progress: bool = True,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Embed texts using a local sentence-transformers model with caching."""
    from sentence_transformers import SentenceTransformer

    texts = filter_invalid_texts(texts)
    cache = load_embedding_cache(cache_dir)
    to_embed = [text for text in texts if text not in cache]

    if not to_embed:
        return cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model, device=device)

    next_chunk_idx = _get_next_chunk_index(cache_dir)
    chunk_embeddings: Dict[str, np.ndarray] = {}

    iterator = range(0, len(to_embed), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Local embeddings")

    for start in iterator:
        batch = to_embed[start : start + batch_size]
        embeddings = model.encode(batch, batch_size=batch_size)
        for text, emb in zip(batch, embeddings):
            cache[text] = np.array(emb, dtype=np.float32)
            chunk_embeddings[text] = cache[text]

    _save_embedding_chunk(cache_dir, chunk_embeddings, next_chunk_idx)
    return cache
