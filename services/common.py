from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


TEXT_LABELS = [
    "restaurant",
    "hotel",
    "attraction",
    "train",
    "taxi",
    "hospital",
    "bus",
    "general",
]

STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "to", "for", "of", "in", "on", "with", "from", "this",
    "that", "use", "using", "please", "help", "need", "most", "near", "into", "your", "user",
    "request", "task", "response", "detailed", "details", "result", "main", "agent", "system",
    "city", "centre", "center", "image", "audio", "text", "book", "classify", "retrieve", "summary",
}


@dataclass
class ModalityOutput:
    modality: str
    available: bool
    summary: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class MissingDependencyError(RuntimeError):
    pass


class DependencyLoader:
    @staticmethod
    def require_transformers():
        try:
            import transformers  # noqa: F401
            return transformers
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyError(
                "transformers is not installed. Run: pip install -r requirements.txt"
            ) from exc



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)



def deterministic_vector(seed_text: str, dim: int = 512) -> np.ndarray:
    digest = hashlib.sha256(seed_text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(0, 1, dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm



def stable_projection(vector: np.ndarray, out_dim: int = 512) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size == out_dim:
        return vector
    key = f"{vector.size}->{out_dim}"
    proj = deterministic_vector(key, dim=vector.size * out_dim).reshape(vector.size, out_dim)
    out = vector @ proj
    out = out.astype(np.float32)
    norm = np.linalg.norm(out)
    return out if norm == 0 else out / norm



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)



def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text



def softmax(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.exp(arr - arr.max())
    total = arr.sum() + 1e-8
    return (arr / total).tolist()



def ensure_path(path_like: str | os.PathLike[str]) -> Path:
    return Path(path_like).expanduser().resolve()



def first_sentence(text: str, max_len: int = 180) -> str:
    text = (text or "").strip()
    if not text:
        return "No textual evidence was available."
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = parts[0].strip()
    return out if len(out) <= max_len else out[: max_len - 3].rstrip() + "..."



def safe_json_dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)



def keyword_set(text: str, limit: int = 8) -> List[str]:
    clean = normalize_text(text)
    tokens = re.findall(r"[a-zA-Z0-9]+", clean)
    out: List[str] = []
    for token in tokens:
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in out:
            out.append(token)
        if len(out) >= limit:
            break
    return out
