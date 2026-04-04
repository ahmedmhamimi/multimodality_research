from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .common import (
    DependencyLoader,
    ModalityOutput,
    deterministic_vector,
    ensure_path,
    first_sentence,
    stable_projection,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


class AudioService:
    def __init__(self, config: Any):
        self.config = config
        self.device = config.DEVICE
        self.processor = None
        self.model = None
        self.mode = "heuristic"
        self._load()

    def _load(self) -> None:
        model_path = ensure_path(self.config.AUDIO_MODEL_PATH)
        checkpoint_path = ensure_path(self.config.AUDIO_CHECKPOINT_PATH)
        if not model_path.exists() and not self.config.ENABLE_REMOTE_MODEL_DOWNLOAD:
            return
        try:
            DependencyLoader.require_transformers()
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            load_target = str(model_path) if model_path.exists() else self.config.DEFAULT_AUDIO_MODEL_NAME
            self.processor = Wav2Vec2Processor.from_pretrained(load_target)
            self.model = Wav2Vec2ForCTC.from_pretrained(load_target)
            if checkpoint_path.exists() and torch is not None:
                state = torch.load(str(checkpoint_path), map_location=self.device)
                if isinstance(state, dict):
                    self.model.load_state_dict(state, strict=False)
            if torch is not None:
                self.model.to(self.device)
                self.model.eval()
            self.mode = "wav2vec2"
        except Exception:
            self.processor = None
            self.model = None
            self.mode = "heuristic"

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        if sf is None:
            raise RuntimeError("soundfile is required to read audio files.")
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform.astype(np.float32), int(sr)

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not audio_path:
            return {"text": "", "confidence": 0.0, "mode": self.mode}
        filename_text = Path(audio_path).stem.replace("_", " ").replace("-", " ")
        if self.mode == "wav2vec2" and self.processor is not None and self.model is not None and torch is not None and sf is not None:
            waveform, sr = self._load_audio(audio_path)
            inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                pred_ids = torch.argmax(outputs.logits, dim=-1)
                text = self.processor.batch_decode(pred_ids)[0]
                confidence = float(torch.softmax(outputs.logits, dim=-1).max(dim=-1).values.mean().cpu().item())
                return {"text": text.strip(), "confidence": confidence, "mode": self.mode, "hidden_states": outputs.hidden_states}
        return {
            "text": f"Fallback transcription based on file name: {filename_text}",
            "confidence": 0.25,
            "mode": self.mode,
        }

    def encode(self, audio_path: str, transcript: str | None = None) -> np.ndarray:
        if not audio_path:
            return np.zeros(self.config.FUSION_OUTPUT_DIM, dtype=np.float32)
        if self.mode == "wav2vec2" and self.processor is not None and self.model is not None and torch is not None and sf is not None:
            waveform, sr = self._load_audio(audio_path)
            inputs = self.processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
                return stable_projection(hidden, out_dim=self.config.FUSION_OUTPUT_DIM)
        seed_text = transcript or Path(audio_path).stem
        return deterministic_vector(seed_text, dim=self.config.FUSION_OUTPUT_DIM)

    def analyze(self, audio_path: str) -> ModalityOutput:
        if not audio_path:
            return ModalityOutput("audio", False, "No audio provided.", np.zeros(self.config.FUSION_OUTPUT_DIM, dtype=np.float32), {})
        transcription = self.transcribe(audio_path)
        transcript = transcription.get("text", "")
        embedding = self.encode(audio_path, transcript=transcript)
        summary = first_sentence(transcript) if transcript else "Audio was provided but no transcript was produced."
        return ModalityOutput(
            modality="audio",
            available=True,
            summary=summary,
            embedding=embedding,
            metadata={
                "transcript": transcript,
                "confidence": transcription.get("confidence", 0.0),
                "mode": transcription.get("mode", self.mode),
                "file_name": Path(audio_path).name,
            },
        )
