# Methodology

## 1. System Overview

This study implements a task-oriented multimodal information retrieval agent as a modular Flask-based system. The architecture contains three modality-specific processing branches for text, image, and audio. Each branch produces a structured output that includes a semantic embedding, a lightweight modality summary, and modality-specific metadata. These outputs are forwarded to a fusion layer that integrates the representations and passes the fused signal to a task completion module.

The system is designed to align with the project roadmap by combining independent modality pipelines, multimodal fusion mechanisms, and task-oriented reasoning in one reproducible framework. The implementation follows the proposed direction of using BERT for text understanding, CLIP for image representation, and Wav2Vec2 for speech processing, then combining them through concatenation, cross-modal attention, and late fusion.

## 2. Modality-Specific Encoders

### 2.1 Text Encoder

The text branch is built around a BERT-style sequence classification model fine-tuned on MultiWOZ-style service labels. The input query is cleaned, normalized, and truncated to a fixed maximum length. The encoder performs two functions. First, it predicts a task-relevant service label such as restaurant, hotel, taxi, or attraction. Second, it extracts a dense text representation from the final hidden layer. The [CLS] token embedding is projected into a fixed 512-dimensional fusion space. This branch therefore contributes both task intent evidence and semantic text features.

### 2.2 Image Encoder

The image branch uses a CLIP-style visual encoder derived from the fine-tuned image notebook. An input image is converted to RGB, resized to the CLIP input resolution, normalized, and encoded into a dense visual representation. The image branch also computes prompt-based similarity scores against a small label bank so that the system can infer whether the image is more consistent with domains such as restaurant, hotel, attraction, transport, or hospital. The final image embedding is projected into the same 512-dimensional fusion space used by the text branch.

### 2.3 Audio Encoder

The audio branch is based on a Wav2Vec2 connectionist temporal classification pipeline. The waveform is loaded, resampled when required, and converted into model input features through the processor used during fine-tuning. The model then generates two outputs. First, it decodes the most likely transcript. Second, it extracts a hidden-state representation that is mean pooled and projected into the common 512-dimensional fusion space. The transcript is treated as explicit linguistic evidence, while the pooled hidden representation is used for embedding-level fusion.

## 3. Multimodal Fusion Strategies

Three fusion strategies are implemented so that the comparative effect of fusion design can be measured directly.

### 3.1 Concatenation Fusion

In concatenation fusion, the system stacks the available modality embeddings into one long vector and then projects the result into a fixed 512-dimensional fused representation. This method preserves modality-specific information explicitly and serves as a strong baseline because of its simplicity and stable behavior.

### 3.2 Cross-Modal Attention Fusion

In cross-modal attention fusion, each modality embedding is treated as a modality token. A scaled dot-product attention matrix is computed across the modality tokens. The attended states are averaged to create the fused representation. This design allows the model to adaptively emphasize modalities that are more relevant to the current task. For example, when a user query contains weak text but highly informative audio or image evidence, the attention mechanism can shift more weight toward those modalities.

### 3.3 Late Fusion

In late fusion, each modality first produces an independent prediction or confidence score. These modality-level outputs are then combined at decision time through confidence-based weighting. This method is useful when the modalities have different reliability levels, such as a strong speech transcript but a noisy image. Late fusion also provides strong interpretability because the contribution of each modality remains visible until the final decision step.

## 4. Task Completion Logic

After fusion, the system forwards the fused representation and modality metadata to a task completion module. The task completion logic has two stages.

The first stage is task selection. The system either accepts a user-selected task or infers the most likely task automatically from the textual request. The supported tasks are detailed response generation, domain classification, retrieval summary, and simulated task execution.

The second stage is action construction. The module inspects the modality evidence and the fusion output, identifies the dominant domain, and produces a structured response. For classification, the system returns the predicted domain and modality evidence. For retrieval summary, it returns the most relevant multimodal evidence. For task execution, it produces a safe execution plan that shows what a real deployment would send to external APIs such as booking, routing, or notification services. This design keeps the Week 5 implementation complete while avoiding unsafe or unsupported real-world actions.

## 5. Evaluation Protocol

The evaluation protocol measures whether multimodal fusion improves task completion relative to single-stream behavior. The system evaluates each case by running the full pipeline and computing the following metrics.

### 5.1 Task Success Rate

Task success rate measures whether the system completes the intended task for a test case. A case is marked as successful when the predicted task matches the expected task or when the returned response covers the required evidence with acceptable coverage.

### 5.2 Precision, Recall, and F1

Precision and recall are computed over expected evidence keywords defined in the evaluation set. Precision measures how much of the generated evidence is relevant, while recall measures how much of the required evidence is recovered. F1 summarizes the balance between both metrics.

### 5.3 Task Completion Time

Task completion time is measured as the elapsed runtime of the end-to-end pipeline for one case. This metric is important because a task-oriented agent must not only be correct but also responsive.

### 5.4 Cross-Modal Relevance

Cross-modal relevance measures how well the system selected the expected modalities for each case. This score is computed by comparing the modalities used by the agent with the modalities that are marked as relevant in the evaluation case.

## 6. Week 5 Experimental Output

The Week 5 system therefore delivers four concrete outputs. First, it provides implemented fusion methods. Second, it provides explicit task completion logic. Third, it supports initial multimodal testing through a Flask interface and a JSON evaluation endpoint. Fourth, it generates a reproducible methodology section that can be inserted directly into the paper.

## 7. Reproducibility

To support reproducibility, the implementation separates configuration from code and allows the user to attach the best saved models from the training notebooks through environment variables. This means the same Flask system can run in two modes. In fallback mode, it executes the full pipeline with deterministic surrogate embeddings for interface testing. In research mode, it loads the fine-tuned BERT, CLIP, and Wav2Vec2 artifacts and performs the intended multimodal experiments.
