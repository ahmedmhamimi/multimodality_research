from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from config import Config
from services.orchestrator import MultimodalOrchestrator


app = Flask(__name__)
app.config.from_object(Config)
orchestrator = MultimodalOrchestrator(Config)


def _request_spec_from_form(user_text: str, requested_task: str, image_path: str | None, audio_path: str | None) -> dict:
    expected_keywords_raw = request.form.get("expected_keywords", "")
    expected_keywords = [item.strip() for item in expected_keywords_raw.split(",") if item.strip()]
    expected_modalities = ["text"] if user_text.strip() else []
    if image_path:
        expected_modalities.append("image")
    if audio_path:
        expected_modalities.append("audio")
    return {
        "title": request.form.get("case_title", "Interactive user request") or "Interactive user request",
        "text": user_text,
        "expected_task": request.form.get("expected_task") or (requested_task if requested_task != "auto" else ""),
        "expected_domain": request.form.get("expected_domain", "").strip(),
        "expected_keywords": expected_keywords,
        "expected_modalities": expected_modalities,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        fusion_methods=app.config["FUSION_METHODS"],
        task_options=app.config["TASK_OPTIONS"],
        model_status=orchestrator.model_status(),
    )


@app.route("/run", methods=["POST"])
def run_pipeline():
    user_text = request.form.get("user_text", "")
    fusion_method = request.form.get("fusion_method", "concatenation")
    requested_task = request.form.get("task_name", "auto")

    image_path = orchestrator.save_upload(request.files.get("image_file"))
    audio_path = orchestrator.save_upload(request.files.get("audio_file"))
    request_spec = _request_spec_from_form(user_text, requested_task, image_path, audio_path)

    result = orchestrator.run(
        user_text=user_text,
        image_path=image_path,
        audio_path=audio_path,
        fusion_method=fusion_method,
        requested_task=requested_task,
        request_spec=request_spec,
    )
    return render_template(
        "result.html",
        result=result,
        user_text=user_text,
        fusion_method=fusion_method,
        requested_task=requested_task,
    )


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(force=True, silent=True) or {}
    request_spec = {
        "title": payload.get("case_title", "API user request"),
        "text": payload.get("user_text", ""),
        "expected_task": payload.get("expected_task") or payload.get("task_name", ""),
        "expected_domain": payload.get("expected_domain", ""),
        "expected_keywords": payload.get("expected_keywords", []),
        "expected_modalities": payload.get("expected_modalities", []),
    }
    result = orchestrator.run(
        user_text=payload.get("user_text", ""),
        image_path=payload.get("image_path"),
        audio_path=payload.get("audio_path"),
        fusion_method=payload.get("fusion_method", "concatenation"),
        requested_task=payload.get("task_name", "auto"),
        request_spec=request_spec,
    )
    return jsonify(result)


@app.route("/methodology")
def methodology():
    methodology_path = Path(app.root_path) / "methodology_section.md"
    content = methodology_path.read_text(encoding="utf-8")
    return render_template("methodology.html", content=content)


@app.route("/evaluation")
def evaluation():
    fusion_method = request.args.get("fusion_method", "concatenation")
    report = orchestrator.evaluation_service.run_batch(fusion_method=fusion_method)
    return render_template(
        "evaluation.html",
        report=report,
        fusion_methods=app.config["FUSION_METHODS"],
    )


@app.route("/api/evaluation")
def api_evaluation():
    fusion_method = request.args.get("fusion_method", "concatenation")
    report = orchestrator.evaluation_service.run_batch(fusion_method=fusion_method)
    return jsonify(report)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "week5_multimodal_flask_system"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
