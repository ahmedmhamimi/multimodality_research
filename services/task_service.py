from __future__ import annotations

from typing import Any, Dict, List

from .common import ModalityOutput, normalize_text


class TaskService:
    def __init__(self, config: Any):
        self.config = config

    def infer_task(self, requested_task: str, user_text: str, modality_outputs: List[ModalityOutput]) -> str:
        if requested_task and requested_task != "auto":
            return requested_task
        text = normalize_text(user_text)
        if any(token in text for token in ["book", "reserve", "schedule", "call", "execute", "do this"]):
            return "task_execution"
        if any(token in text for token in ["classify", "intent", "domain", "label"]):
            return "classification"
        if any(token in text for token in ["retrieve", "match", "similar"]):
            return "retrieval_summary"
        return "detailed_response"

    def execute(
        self,
        task_name: str,
        user_text: str,
        modality_outputs: List[ModalityOutput],
        fusion_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence = {o.modality: o for o in modality_outputs if o.available}
        text_output = evidence.get("text")
        image_output = evidence.get("image")
        audio_output = evidence.get("audio")

        dominant_domain = "general"
        if text_output:
            dominant_domain = text_output.metadata.get("classification", {}).get("label", "general")
        elif image_output:
            dominant_domain = image_output.metadata.get("label", "general")

        base_lines = []
        if text_output:
            base_lines.append(f"Text evidence suggests the main domain is {text_output.metadata.get('classification', {}).get('label', 'general')}.")
        if image_output:
            image_label = image_output.metadata.get("label", image_output.metadata.get("summary", "general"))
            base_lines.append(f"Image evidence points to {image_label}.")
        if audio_output:
            transcript = audio_output.metadata.get("transcript", "")
            base_lines.append(f"Audio transcription says: {transcript[:140]}")

        if task_name == "classification":
            response = {
                "predicted_domain": dominant_domain,
                "task_success_hint": f"The system mapped the request to the {dominant_domain} domain.",
                "response_text": " ".join(base_lines) or "No modality evidence was available for classification.",
                "action_plan": ["Return the predicted domain", "Expose confidence by modality", "Store result for evaluation"],
            }
        elif task_name == "task_execution":
            response = {
                "predicted_domain": dominant_domain,
                "task_success_hint": "The agent prepared an execution plan. External APIs are not connected, so this is a simulated action.",
                "response_text": (
                    "The agent used the fused modalities to prepare a task plan. "
                    + " ".join(base_lines)
                    + " A real deployment can replace this step with live booking, routing, or notification APIs."
                ).strip(),
                "action_plan": [
                    "Validate the user goal",
                    "Select the dominant modality or fusion output",
                    "Generate parameters for the target action",
                    "Return a safe execution preview",
                ],
            }
        elif task_name == "retrieval_summary":
            response = {
                "predicted_domain": dominant_domain,
                "task_success_hint": "The agent returned a fused retrieval summary.",
                "response_text": " ".join(base_lines) or "No modalities were available for retrieval summarization.",
                "action_plan": ["Collect modality evidence", "Fuse representations", "Summarize the most relevant evidence"],
            }
        else:
            response = {
                "predicted_domain": dominant_domain,
                "task_success_hint": "The agent produced a detailed multimodal response.",
                "response_text": (
                    "The agent integrated all available modalities and produced a consolidated answer. "
                    + " ".join(base_lines)
                    + f" Fusion agreement was {fusion_result.get('agreement', 0.0):.3f}."
                ).strip(),
                "action_plan": ["Parse each modality", "Fuse embeddings", "Generate a grounded response"],
            }

        response["task_name"] = task_name
        response["used_modalities"] = sorted(evidence.keys())
        response["fusion_method"] = fusion_result.get("method")
        response["fusion_weights"] = fusion_result.get("weights", {})
        return response
