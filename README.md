---
title: Multimodal Content Moderation
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: mit
short_description: BERT + ViT moderation for text and images
---

# Multimodal Content Moderation

Detects harmful content in **text** and **images** and flags it for review.
It has a web UI and a REST API served by the same FastAPI app.

| Input | Model | What it scores |
|---|---|---|
| Text | [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) (BERT, fine-tuned on Jigsaw) | toxic, severe toxicity, obscene, threat, insult, identity hate |
| Image | [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection) (ViT) | NSFW |
| Image | [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) (ViT) | likelihood the image is AI-generated (shown as information only; it does not flag content) |

## How it decides

1. Text is scored by BERT as a **multi-label** problem (sigmoid per category), so a comment can be both an insult and a threat.
2. Images are converted to RGB (transparent, palette and animated images handled) and scored by the ViT classifiers.
3. **Late fusion:** text and image category scores are combined into one result. Content is **flagged when any harm category reaches the threshold**. A lower threshold is stricter.

```
text  ──► BERT toxicity (6 labels) ──┐
                                      ├──► combine scores ──► flagged? + top category
image ──► ViT NSFW ──────────────────┘
image ──► ViT AI-generated detector ───► informational score
```

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 for the UI and http://localhost:7860/docs for the API docs.
The first start downloads about 1 GB of model weights.

## REST API

| Method | Path | Body |
|---|---|---|
| GET | `/api/health` | |
| POST | `/api/moderate/text` | JSON `{"text": "...", "threshold": 0.5}` |
| POST | `/api/moderate/batch` | JSON `{"texts": ["...", "..."], "threshold": 0.5}` (up to 64) |
| POST | `/api/moderate/image` | multipart: `image` file, optional `threshold` |
| POST | `/api/moderate` | multipart: optional `text`, optional `image`, optional `threshold` |

```bash
curl -X POST http://localhost:7860/api/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an idiot"}'

curl -X POST http://localhost:7860/api/moderate \
  -F "text=look at this" -F "image=@photo.jpg"
```

Example response:

```json
{
  "flagged": true,
  "verdict": "Flagged: Toxic (97%)",
  "top_category": "Toxic",
  "top_score": 0.97,
  "categories": {"Toxic": 0.97, "Insult": 0.91, "Threat": 0.01, "...": 0.0},
  "modalities": ["text"],
  "threshold": 0.5,
  "latency_ms": 41.3,
  "ai_generated": null,
  "notes": []
}
```

## Evaluation

```bash
pip install -r requirements-dev.txt
python eval/evaluate.py
```

This measures text accuracy on a balanced, fixed-seed sample of the public
[Civil Comments](https://huggingface.co/datasets/google/civil_comments) test set
and per-request latency for text, image and combined requests. Results are written
to `eval/results.md`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use lightweight stand-ins for the models, so they run in about a second
without downloading anything. They cover scoring rules, thresholds, input validation,
image format handling, multimodal fusion and every API endpoint.

## Project layout

```
app.py                  entry point: FastAPI + Gradio UI on one server (port 7860)
moderation/
  engine.py             model loading, scoring, fusion
  api.py                REST endpoints
  ui.py                 Gradio interface
  config.py             model names and limits (overridable with env vars)
eval/evaluate.py        accuracy and latency measurement
tests/                  pytest suite
```

## Configuration

Set these environment variables to change behaviour (for example in the Space's settings):

| Variable | Default |
|---|---|
| `TEXT_MODEL` | `unitary/toxic-bert` |
| `NSFW_IMAGE_MODEL` | `Falconsai/nsfw_image_detection` |
| `AI_IMAGE_MODEL` | `umm-maybe/AI-image-detector` (set to empty to disable) |
| `DEFAULT_THRESHOLD` | `0.5` |
| `MAX_TEXT_CHARS` | `5000` |

## Limitations

- English text only; toxicity models can over-flag identity terms and miss sarcasm or coded language.
- The NSFW model covers sexual content, not violence or gore.
- The AI-image detector was trained on 2022-era generators and is unreliable on newer ones, which is why it does not flag content.
- Meant for research and demos, not as the only safeguard on a real platform.
