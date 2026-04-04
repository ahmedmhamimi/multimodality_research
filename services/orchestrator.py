from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .audio_service import AudioService
from .evaluation_service import EvaluationService
from .fusion_service import FusionService
from .image_service import ImageService
from .task_service import TaskService
from .text_service import TextService


class MultimodalOrchestrator:
    def __init__(self, config: Any):
        self.config = config
        self.text_service = TextService(config)
        self.image_service = ImageService(config)
        self.audio_service = AudioService(config)
        self.fusion_service = FusionService(config)
        self.task_service = TaskService(config)
        self.evaluation_service = EvaluationService(config, self)
        Path(self.config.UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

    def model_status(self) -> Dict[str, Dict[str, str]]:
        return {
            "text": {"mode": self.text_service.mode, "path": str(self.config.TEXT_MODEL_PATH)},
            "image": {"mode": self.image_service.mode, "path": str(self.config.IMAGE_MODEL_PATH)},
            "audio": {"mode": self.audio_service.mode, "path": str(self.config.AUDIO_CHECKPOINT_PATH)},
        }

    def save_upload(self, file_storage: Any) -> Optional[str]:
        if file_storage is None or not getattr(file_storage, "filename", None):
            return None
        target = Path(self.config.UPLOAD_FOLDER) / Path(file_storage.filename).name
        file_storage.save(str(target))
        return str(target)

    def run(
        self,
        user_text: str,
        image_path: Optional[str],
        audio_path: Optional[str],
        fusion_method: str,
        requested_task: str,
        request_spec: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text_output = self.text_service.analyze(user_text)
        image_output = self.image_service.analyze(image_path)
        audio_output = self.audio_service.analyze(audio_path)
        fusion_result = self.fusion_service.fuse([text_output, image_output, audio_output], method=fusion_method)
        predicted_task = self.task_service.infer_task(requested_task, user_text, [text_output, image_output, audio_output])
        task_result = self.task_service.execute(predicted_task, user_text, [text_output, image_output, audio_output], fusion_result)
        result = {
            "predicted_task": predicted_task,
            "modalities": {
                "text": {
                    "summary": text_output.summary,
                    "metadata": text_output.metadata,
                    "available": text_output.available,
                },
                "image": {
                    "summary": image_output.summary,
                    "metadata": image_output.metadata,
                    "available": image_output.available,
                },
                "audio": {
                    "summary": audio_output.summary,
                    "metadata": audio_output.metadata,
                    "available": audio_output.available,
                },
            },
            "fusion": {
                "method": fusion_result.get("method"),
                "weights": fusion_result.get("weights"),
                "agreement": fusion_result.get("agreement"),
                "attention_matrix": fusion_result.get("attention_matrix"),
            },
            "task_result": task_result,
            "request": {
                "user_text": user_text,
                "image_path": image_path,
                "audio_path": audio_path,
                "requested_task": requested_task,
            },
        }
        spec = request_spec or {}
        spec.setdefault("text", user_text)
        if spec:
            result["evaluation"] = self.evaluation_service.evaluate_single(result, spec)
        return result

    def run_from_case(self, case: Dict[str, Any], fusion_method: str) -> Dict[str, Any]:
        return self.run(
            user_text=case.get("text", ""),
            image_path=case.get("image_path"),
            audio_path=case.get("audio_path"),
            fusion_method=fusion_method,
            requested_task=case.get("expected_task", "auto"),
            request_spec=case,
        )
