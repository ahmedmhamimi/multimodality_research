from __future__ import annotations

from typing import Any, Dict

import numpy as np
from PIL import Image

from .common import DependencyLoader, ModalityOutput, ensure_path, stable_projection

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class ImageService:
    PROMPTS = [
        "a restaurant scene",
        "a hotel room",
        "a tourist attraction",
        "a train station",
        "a taxi or car ride",
        "a hospital or clinic",
        "a bus or bus stop",
        "a general everyday scene",
    ]
    LABELS = ["restaurant", "hotel", "attraction", "train", "taxi", "hospital", "bus", "general"]

    def __init__(self, config: Any):
        self.config = config
        self.device = config.DEVICE
        self.processor = None
        self.model = None
        self.mode = "heuristic"
        self.load_source = "heuristic"
        self._load()

    def _load(self) -> None:
        image_model_path = ensure_path(self.config.IMAGE_MODEL_PATH)
        if not image_model_path.exists() and not self.config.ENABLE_REMOTE_MODEL_DOWNLOAD:
            return
        try:
            DependencyLoader.require_transformers()
            from transformers import CLIPModel, CLIPProcessor

            if image_model_path.exists() and image_model_path.suffix == ".pt":
                ckpt = torch.load(str(image_model_path), map_location=self.device) if torch is not None else {}
                clip_name = self.config.DEFAULT_IMAGE_MODEL_NAME
                if isinstance(ckpt, dict):
                    clip_name = ckpt.get("clip_model_name", clip_name)
                self.processor = CLIPProcessor.from_pretrained(clip_name)
                self.model = CLIPModel.from_pretrained(clip_name)
                if isinstance(ckpt, dict):
                    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
                        self.model.load_state_dict(ckpt["model_state"], strict=False)
                    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                        self.model.load_state_dict(ckpt["state_dict"], strict=False)
                self.load_source = str(image_model_path.name)
            else:
                load_target = str(image_model_path) if image_model_path.exists() else self.config.DEFAULT_IMAGE_MODEL_NAME
                self.processor = CLIPProcessor.from_pretrained(load_target)
                self.model = CLIPModel.from_pretrained(load_target)
                self.load_source = load_target
            if torch is not None:
                self.model.to(self.device)
                self.model.eval()
            self.mode = "clip"
        except Exception:
            self.processor = None
            self.model = None
            self.mode = "heuristic"
            self.load_source = "heuristic"

    def _heuristic_summary(self, image: Image.Image, path: str) -> Dict[str, Any]:
        arr = np.asarray(image.resize((64, 64))).astype(np.float32)
        brightness = float(arr.mean() / 255.0)
        width, height = image.size
        aspect_ratio = round(width / max(height, 1), 3)
        label = "general"
        if width > height * 1.2:
            label = "attraction"
        elif brightness < 0.25:
            label = "hotel"
        summary = f"Image size is {width}x{height} with brightness {brightness:.2f} and aspect ratio {aspect_ratio}."
        return {
            "label": label,
            "confidence": 0.45,
            "summary": summary,
            "brightness": brightness,
            "aspect_ratio": aspect_ratio,
            "path": path,
        }

    def _extract_feature_tensor(self, output):
        if torch is None:
            raise RuntimeError("Torch is not available.")
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state.mean(dim=1)
        raise TypeError(f"Unsupported image feature output type: {type(output)}")

    def encode(self, image_path: str) -> np.ndarray:
        if not image_path:
            return np.zeros(self.config.FUSION_OUTPUT_DIM, dtype=np.float32)
        image = Image.open(image_path).convert("RGB")
        if self.mode == "clip" and self.model is not None and self.processor is not None and torch is not None:
            try:
                with torch.no_grad():
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    raw_output = self.model.get_image_features(**inputs)
                    feat_tensor = self._extract_feature_tensor(raw_output)
                    feats = feat_tensor.squeeze(0).detach().cpu().numpy()
                    return stable_projection(feats, out_dim=self.config.FUSION_OUTPUT_DIM)
            except Exception:
                pass
        arr = np.asarray(image.resize((32, 32))).astype(np.float32).reshape(-1)
        return stable_projection(arr, out_dim=self.config.FUSION_OUTPUT_DIM)

    def analyze(self, image_path: str) -> ModalityOutput:
        if not image_path:
            return ModalityOutput("image", False, "No image provided.", np.zeros(self.config.FUSION_OUTPUT_DIM, dtype=np.float32), {})
        image = Image.open(image_path).convert("RGB")
        embedding = self.encode(image_path)

        if self.mode == "clip" and self.model is not None and self.processor is not None and torch is not None:
            try:
                with torch.no_grad():
                    inputs = self.processor(text=self.PROMPTS, images=image, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    logits = outputs.logits_per_image.squeeze(0).detach().cpu().numpy()
                    probs = np.exp(logits - logits.max())
                    probs = probs / (probs.sum() + 1e-8)
                    best_idx = int(np.argmax(probs))
                    label = self.LABELS[best_idx]
                    summary = f"The image is most aligned with '{label}' and the top prompt score is {probs[best_idx]:.3f}."
                    metadata = {
                        "label": label,
                        "confidence": float(probs[best_idx]),
                        "prompt_scores": {self.LABELS[i]: float(probs[i]) for i in range(len(self.LABELS))},
                        "mode": self.mode,
                        "size": {"width": image.size[0], "height": image.size[1]},
                        "model_source": self.load_source,
                    }
            except Exception:
                heuristic = self._heuristic_summary(image, image_path)
                summary = heuristic["summary"]
                metadata = {**heuristic, "mode": "heuristic", "model_source": self.load_source}
        else:
            heuristic = self._heuristic_summary(image, image_path)
            summary = heuristic["summary"]
            metadata = {**heuristic, "mode": self.mode, "model_source": self.load_source}

        return ModalityOutput(
            modality="image",
            available=True,
            summary=summary,
            embedding=embedding,
            metadata=metadata,
        )
