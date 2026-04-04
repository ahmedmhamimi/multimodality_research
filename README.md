# Week 5 Multimodal Fusion Flask System

This project is a runnable Flask system for Week 5 of your task-oriented multimodal IR agent.

It combines
- text understanding from a BERT-style service classifier
- image encoding from a CLIP-style encoder
- audio transcription and representation from a Wav2Vec2-style speech model
- three fusion strategies
- task completion logic
- per-request evaluation for task success rate, precision, recall, F1, task time, modality coverage, and cross-modal relevance
- a batch evaluation page for stored benchmark cases

## What changed in this updated version

- fixed the Flask config wiring so the app uses `Config` correctly
- fixed the image branch so CLIP outputs that return `BaseModelOutputWithPooling` no longer crash the app
- simplified the web flow so one request returns both the response and the evaluation on the same result page
- added optional evaluation fields on the main form so you can define expected task, expected domain, and expected keywords for the current request
- kept the separate evaluation page for stored benchmark cases only

## Project structure

```text
week5_multimodal_flask_system/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── methodology_section.md
├── methodology_section.docx
├── services/
├── templates/
├── static/
├── data/
├── models/
└── uploads/
```

## How to plug in your best notebook models

Put your best models in the `models` folder or point to them with environment variables.

### Text model
Your text notebook saves a fine-tuned BERT classifier.

Set
```bash
export TEXT_MODEL_PATH=/absolute/path/to/bert_multiwoz_best
```

### Image model
Your image notebook saves a `.pt` CLIP checkpoint.

Set
```bash
export IMAGE_MODEL_PATH=/absolute/path/to/sir_yes_sir.pt
```

### Audio model
Your audio notebook uses a Wav2Vec2 base directory and a separate best checkpoint.

Set
```bash
export AUDIO_MODEL_PATH=/absolute/path/to/wav2vec2-base
export AUDIO_CHECKPOINT_PATH=/absolute/path/to/best_model.pt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
python app.py
```

Open
```text
http://127.0.0.1:5000
```

## Routes

- `/` main interface with inference plus per-request evaluation
- `/methodology` paper methodology section
- `/evaluation` batch evaluation dashboard for stored cases
- `/api/run` JSON inference endpoint with inline evaluation
- `/api/evaluation` JSON batch evaluation endpoint
- `/health` health check

## Fusion methods

### 1. Concatenation
The system stacks modality embeddings then projects them back to one fused vector.

### 2. Cross attention
The system forms a modality-by-modality attention matrix and averages attended modality states.

### 3. Late fusion
The system keeps each modality decision separate, then combines them with confidence-based weights.

## Task logic

The task module supports
- auto task inference
- detailed multimodal response
- classification
- retrieval summary
- simulated task execution

A real deployment can connect these outputs to live booking systems, search APIs, or workflow engines.

## Evaluation

### Per-request evaluation
Every run now returns
- task success rate
- precision
- recall
- F1
- average task time
- cross-modal relevance
- modality coverage

These metrics are attached to the exact request you just submitted.

### Batch evaluation
The stored benchmark cases in `data/sample_eval_cases.json` are used by `/evaluation`.

## Notes

- If the real models are not available, the app still runs in fallback mode.
- Fallback mode uses deterministic embeddings and heuristic logic so the interface, fusion layer, evaluation flow, and methodology still work end to end.
- Replace fallback mode with your saved notebook models for thesis-grade experiments.
