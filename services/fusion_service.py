from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .common import ModalityOutput, cosine_similarity, deterministic_vector, stable_projection


class FusionService:
    def __init__(self, config: Any):
        self.config = config

    def fuse(self, outputs: List[ModalityOutput], method: str = "concatenation") -> Dict[str, Any]:
        available = [o for o in outputs if o.available]
        if not available:
            zero = np.zeros(self.config.FUSION_OUTPUT_DIM, dtype=np.float32)
            return {"method": method, "vector": zero, "weights": {}, "agreement": 0.0}

        if method == "cross_attention":
            return self._cross_attention(available)
        if method == "late_fusion":
            return self._late_fusion(available)
        return self._concatenation(available)

    def _concatenation(self, outputs: List[ModalityOutput]) -> Dict[str, Any]:
        concat = np.concatenate([o.embedding for o in outputs], axis=0)
        fused = stable_projection(concat, out_dim=self.config.FUSION_OUTPUT_DIM)
        weights = {o.modality: round(1.0 / len(outputs), 3) for o in outputs}
        agreement = self._agreement(outputs)
        return {"method": "concatenation", "vector": fused, "weights": weights, "agreement": agreement}

    def _cross_attention(self, outputs: List[ModalityOutput]) -> Dict[str, Any]:
        matrix = np.stack([o.embedding for o in outputs], axis=0)
        attn_scores = matrix @ matrix.T / np.sqrt(matrix.shape[1])
        attn_scores = np.exp(attn_scores - attn_scores.max(axis=1, keepdims=True))
        attn_weights = attn_scores / (attn_scores.sum(axis=1, keepdims=True) + 1e-8)
        attended = attn_weights @ matrix
        fused = attended.mean(axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        modality_weights = {outputs[i].modality: float(attn_weights.mean(axis=0)[i]) for i in range(len(outputs))}
        return {
            "method": "cross_attention",
            "vector": fused.astype(np.float32),
            "weights": modality_weights,
            "agreement": self._agreement(outputs),
            "attention_matrix": attn_weights.round(4).tolist(),
        }

    def _late_fusion(self, outputs: List[ModalityOutput]) -> Dict[str, Any]:
        weight_map = {}
        vectors = []
        weights = []
        for output in outputs:
            confidence = 0.5
            if output.modality == "text":
                confidence = float(output.metadata.get("classification", {}).get("confidence", 0.5))
            elif output.modality == "image":
                confidence = float(output.metadata.get("confidence", output.metadata.get("label_confidence", 0.5)))
            elif output.modality == "audio":
                confidence = float(output.metadata.get("confidence", 0.5))
            confidence = max(confidence, 0.15)
            weight_map[output.modality] = round(confidence, 3)
            vectors.append(output.embedding)
            weights.append(confidence)
        w = np.asarray(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        fused = np.average(np.stack(vectors, axis=0), axis=0, weights=w)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return {"method": "late_fusion", "vector": fused.astype(np.float32), "weights": weight_map, "agreement": self._agreement(outputs)}

    def _agreement(self, outputs: List[ModalityOutput]) -> float:
        if len(outputs) <= 1:
            return 1.0
        sims = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sims.append(cosine_similarity(outputs[i].embedding, outputs[j].embedding))
        return float(np.mean(sims)) if sims else 1.0
