from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .common import (
    DependencyLoader,
    ModalityOutput,
    TEXT_LABELS,
    deterministic_vector,
    ensure_path,
    first_sentence,
    normalize_text,
    softmax,
    stable_projection,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class TextService:
    def __init__(self, config: Any):
        self.config = config
        self.device = config.DEVICE
        self.model = None
        self.tokenizer = None
        self.mode = "heuristic"
        self.label_list = TEXT_LABELS
        self._load()

    def _load(self) -> None:
        model_path = ensure_path(self.config.TEXT_MODEL_PATH)
        if not model_path.exists() and not self.config.ENABLE_REMOTE_MODEL_DOWNLOAD:
            return
        try:
            DependencyLoader.require_transformers()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            load_target = str(model_path) if model_path.exists() else self.config.DEFAULT_TEXT_MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(load_target)
            self.model = AutoModelForSequenceClassification.from_pretrained(load_target)
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                self.label_list = [self.model.config.id2label[i] for i in sorted(self.model.config.id2label)]
            if torch is not None:
                self.model.to(self.device)
                self.model.eval()
            self.mode = "transformer"
        except Exception:
            self.model = None
            self.tokenizer = None
            self.mode = "heuristic"

    def classify(self, text: str) -> Dict[str, Any]:
        clean = normalize_text(text)
        if not clean:
            return {"label": "general", "confidence": 0.0, "probabilities": {label: 0.0 for label in self.label_list}}

        if self.mode == "transformer" and self.model is not None and self.tokenizer is not None and torch is not None:
            with torch.no_grad():
                inputs = self.tokenizer(clean, return_tensors="pt", truncation=True, max_length=192)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
                pred_id = int(np.argmax(probs))
                label = self.label_list[pred_id] if pred_id < len(self.label_list) else "general"
                return {
                    "label": label,
                    "confidence": float(probs[pred_id]),
                    "probabilities": {self.label_list[i]: float(probs[i]) for i in range(len(self.label_list))},
                }

        scores = {label: 0.05 for label in self.label_list}
        keyword_map = {
            "restaurant": ["restaurant", "food", "dinner", "lunch", "cheap", "book table"],
            "hotel": ["hotel", "stay", "room", "check in", "accommodation"],
            "attraction": ["museum", "park", "visit", "attraction", "tour"],
            "train": ["train", "station", "platform", "departure", "arrival"],
            "taxi": ["taxi", "ride", "pickup", "dropoff", "driver"],
            "hospital": ["hospital", "doctor", "clinic", "medical"],
            "bus": ["bus", "coach", "terminal", "route"],
            "general": ["help", "information", "question", "details"],
        }
        for label, words in keyword_map.items():
            scores[label] += sum(1 for word in words if word in clean) * 0.3
        vals = list(scores.values())
        probs = softmax(vals)
        prob_map = {label: float(probs[i]) for i, label in enumerate(scores.keys())}
        label = max(prob_map, key=prob_map.get)
        return {"label": label, "confidence": float(prob_map[label]), "probabilities": prob_map}

    def encode(self, text: str) -> np.ndarray:
        clean = normalize_text(text)
        if not clean:
            return np.zeros(self.config.FUSION_OUTPUT_DIM, dtype=np.float32)

        if self.mode == "transformer" and self.model is not None and self.tokenizer is not None and torch is not None:
            with torch.no_grad():
                inputs = self.tokenizer(clean, return_tensors="pt", truncation=True, max_length=192)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                if outputs.hidden_states is not None:
                    cls_vec = outputs.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy()
                else:
                    cls_vec = outputs.logits.squeeze(0).cpu().numpy()
                return stable_projection(cls_vec, out_dim=self.config.FUSION_OUTPUT_DIM)

        return deterministic_vector(clean, dim=self.config.FUSION_OUTPUT_DIM)

    def analyze(self, text: str) -> ModalityOutput:
        classification = self.classify(text)
        embedding = self.encode(text)
        summary = first_sentence(text)
        return ModalityOutput(
            modality="text",
            available=bool(text and text.strip()),
            summary=summary,
            embedding=embedding,
            metadata={
                "classification": classification,
                "mode": self.mode,
                "length": len((text or "").split()),
            },
        )
