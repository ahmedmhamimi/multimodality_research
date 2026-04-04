from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Set

import numpy as np

from .common import keyword_set, normalize_text


class EvaluationService:
    def __init__(self, config: Any, orchestrator: Any):
        self.config = config
        self.orchestrator = orchestrator

    def load_cases(self) -> List[Dict[str, Any]]:
        with open(self.config.EVAL_CASES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _expected_keywords(self, request_spec: Dict[str, Any]) -> List[str]:
        """
        Use only explicitly provided keywords, or fall back to extracting
        from the input text itself. Never mix in task/domain names as keywords
        because they inflate recall artificially.
        """
        explicit = [normalize_text(k) for k in request_spec.get("expected_keywords", []) if normalize_text(k)]
        if explicit:
            return explicit
        # Fall back: extract keywords strictly from the user input text only
        return keyword_set(request_spec.get("text", ""), limit=8)

    def _predicted_keywords(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract predicted keywords only from grounded model outputs:
        - the predicted domain label
        - the audio transcript (actual speech content)
        - the image label
        - the predicted task name
        Do NOT use the response_text because it is a hardcoded template
        and will never reflect the user's input words, making precision
        artificially low.
        """
        parts = [
            result.get("task_result", {}).get("predicted_domain", ""),
            result.get("predicted_task", "").replace("_", " "),
        ]

        # Add image label if image was available
        image_meta = result.get("modalities", {}).get("image", {}).get("metadata", {})
        if image_meta.get("label"):
            parts.append(image_meta["label"])

        # Add audio transcript if audio was available
        audio_meta = result.get("modalities", {}).get("audio", {}).get("metadata", {})
        transcript = audio_meta.get("transcript", "")
        if transcript:
            parts.append(transcript)

        # Add text classification label
        text_meta = result.get("modalities", {}).get("text", {}).get("metadata", {})
        classification_label = text_meta.get("classification", {}).get("label", "")
        if classification_label:
            parts.append(classification_label)

        combined = " ".join(parts)
        return keyword_set(combined, limit=10)

    def _compute_task_success(
        self,
        task_match: bool,
        domain_match: bool | None,
        recall: float,
        expected_task: str,
        expected_keywords: List[str],
        expected_domain: str,
    ) -> bool:
        """
        Task success requires meeting ALL conditions that were actually set.
        A condition that was never set (empty string / None) is skipped,
        not auto-passed.

        Rules:
        - If expected_task was set   → task_match must be True
        - If expected_keywords set   → recall must be >= 0.5
        - If expected_domain was set → domain_match must be True
        - If nothing was set at all  → return None (unevaluable), shown as 'not set'
        """
        conditions_set = 0
        conditions_passed = 0

        if expected_task:
            conditions_set += 1
            if task_match:
                conditions_passed += 1

        if expected_keywords:
            conditions_set += 1
            if recall >= 0.5:
                conditions_passed += 1

        if expected_domain:
            conditions_set += 1
            if domain_match is True:
                conditions_passed += 1

        if conditions_set == 0:
            # Nothing was specified — cannot evaluate, do not auto-pass
            return None

        return conditions_passed == conditions_set

    def evaluate_single(
        self,
        result: Dict[str, Any],
        request_spec: Dict[str, Any],
        elapsed: float | None = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()

        expected_task = normalize_text(request_spec.get("expected_task", "")).replace(" ", "_")
        predicted_task = normalize_text(result.get("predicted_task", "")).replace(" ", "_")

        expected_domain = normalize_text(request_spec.get("expected_domain", ""))
        predicted_domain = normalize_text(result.get("task_result", {}).get("predicted_domain", ""))

        expected_keywords = self._expected_keywords(request_spec)
        predicted_keywords = self._predicted_keywords(result)
        matched_keywords = sorted(set(expected_keywords) & set(predicted_keywords))

        # Precision: of what the system predicted, how much was relevant
        precision = (
            len(matched_keywords) / len(set(predicted_keywords))
            if predicted_keywords
            else 0.0
        )

        # Recall: of what was expected, how much did the system cover
        recall = (
            len(matched_keywords) / len(set(expected_keywords))
            if expected_keywords
            else 0.0
        )

        f1 = (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )

        expected_modalities: Set[str] = set(request_spec.get("expected_modalities", []))
        used_modalities: Set[str] = set(result.get("task_result", {}).get("used_modalities", []))
        modality_coverage = (
            len(expected_modalities & used_modalities) / max(len(expected_modalities), 1)
            if expected_modalities
            else float(len(used_modalities) > 0)
        )

        task_match = bool(expected_task) and predicted_task == expected_task
        domain_match = (predicted_domain == expected_domain) if expected_domain else None

        if elapsed is None:
            elapsed = time.perf_counter() - start

        task_success = self._compute_task_success(
            task_match=task_match,
            domain_match=domain_match,
            recall=recall,
            expected_task=expected_task,
            expected_keywords=expected_keywords,
            expected_domain=expected_domain,
        )

        # Flag trivially perfect scores caused by single-modality input
        used_modality_count = len(used_modalities)
        trivial_agreement = used_modality_count <= 1

        cross_modal_relevance = float(result.get("fusion", {}).get("agreement", 0.0))

        return {
            "title": request_spec.get("title", "Interactive request"),
            "expected": {
                "task": expected_task or "not_set",
                "domain": expected_domain or "not_set",
                "keywords": expected_keywords,
                "modalities": sorted(expected_modalities),
            },
            "observed": {
                "task": predicted_task or "not_set",
                "domain": predicted_domain or "not_set",
                "keywords": predicted_keywords,
                "modalities": sorted(used_modalities),
            },
            "summary": {
                # task_success is now None if nothing was set, float otherwise
                "task_success_rate": (
                    "not_evaluable"
                    if task_success is None
                    else round(1.0 if task_success else 0.0, 4)
                ),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "average_task_time_seconds": round(float(elapsed), 4),
                "cross_modal_relevance": (
                    "trivial_single_modality"
                    if trivial_agreement
                    else round(cross_modal_relevance, 4)
                ),
                "modality_coverage": round(float(modality_coverage), 4),
            },
            "checks": {
                "task_match": task_match,
                "domain_match": domain_match,
                "matched_keywords": matched_keywords,
                "trivial_agreement_warning": trivial_agreement,
            },
        }

    def run_batch(self, fusion_method: str) -> Dict[str, Any]:
        cases = self.load_cases()
        rows = []

        # Only aggregate numeric metrics — skip string sentinels like "not_evaluable"
        aggregates: Dict[str, List[float]] = {key: [] for key in self.config.EXPLAINABLE_METRICS}

        for case in cases:
            start = time.perf_counter()
            result = self.orchestrator.run_from_case(case, fusion_method=fusion_method)
            elapsed = time.perf_counter() - start
            evaluation = self.evaluate_single(result, case, elapsed=elapsed)
            summary = evaluation["summary"]

            # Resolve display value for task_success_rate
            tsr = summary["task_success_rate"]
            tsr_display = tsr if isinstance(tsr, str) else tsr

            # Resolve cross_modal_relevance display
            cmr = summary["cross_modal_relevance"]
            cmr_display = cmr if isinstance(cmr, str) else cmr

            rows.append(
                {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "expected_task": evaluation["expected"]["task"],
                    "predicted_task": evaluation["observed"]["task"],
                    "task_success": tsr_display,
                    "precision": summary["precision"],
                    "recall": summary["recall"],
                    "f1": summary["f1"],
                    "cross_modal_relevance": cmr_display,
                    "modality_coverage": summary["modality_coverage"],
                    "task_time_seconds": summary["average_task_time_seconds"],
                }
            )

            # Only accumulate numeric values into aggregates
            for key in aggregates:
                val = summary.get(key)
                if isinstance(val, float):
                    aggregates[key].append(val)

        agg_summary = {
            key: round(float(np.mean(values)), 4) if values else "not_evaluable"
            for key, values in aggregates.items()
        }
        agg_summary["cases"] = len(rows)

        return {
            "fusion_method": fusion_method,
            "summary": agg_summary,
            "rows": rows,
        }