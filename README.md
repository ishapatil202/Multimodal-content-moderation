# Multimodal Content Moderation

Flags harmful **text** and **images** using a BERT toxicity classifier and Vision Transformer
image classifiers, served through a **FastAPI** REST API and a **Gradio** web UI.

| Input | Model | Detects |
|---|---|---|
| Text | [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) (BERT) | toxic, severe toxicity, obscene, threat, insult, identity hate |
| Image | [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection) (ViT) | NSFW content |
| Image | [`umm-maybe/AI-image-detector`](https://huggingface.co/umm-maybe/AI-image-detector) (ViT) | AI-generated likelihood (informational only) |

Text and image scores are combined (late fusion), and content is flagged when any harm
category reaches the threshold.

## Results

Measured with `eval/evaluate.py` on 2,000 balanced comments from the
[Civil Comments](https://huggingface.co/datasets/google/civil_comments) test set.
The threshold was tuned on the separate validation split.

| Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| 0.5 (default) | 70.2% | 95.1% | 42.7% | 58.9% | 0.919 |
| 0.05 (tuned) | 83.9% | 85.8% | 81.3% | 83.5% | 0.919 |

Median latency on a MacBook Pro (CPU only): **57 ms** text, **233 ms** image, **295 ms** text + image.
Full results: [`eval/results.md`](eval/results.md).

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

UI at http://localhost:7860, API docs at http://localhost:7860/docs.
The first run downloads about 1 GB of model weights.

## API

| Endpoint | Input |
|---|---|
| `POST /api/moderate/text` | JSON `{"text": "...", "threshold": 0.5}` |
| `POST /api/moderate/batch` | JSON `{"texts": [...]}` (up to 64) |
| `POST /api/moderate/image` | form-data `image` file |
| `POST /api/moderate` | form-data `text` and/or `image` |
| `GET /api/health` | |

```bash
curl -X POST http://localhost:7860/api/moderate/text \
  -H "Content-Type: application/json" -d '{"text": "You are an idiot"}'
# {"flagged": true, "verdict": "Flagged: Toxic (99%)", "categories": {...}, "latency_ms": 57, ...}
```

## Tests and evaluation

```bash
pip install -r requirements-dev.txt
pytest                      # 43 tests, no model download needed
python eval/evaluate.py     # accuracy + latency, writes eval/results.md
```

## Limitations

English text only. The NSFW model does not cover violence or gore. The AI-image detector
predates newer image generators, so it is shown as information and never flags content.
