"""
Moderation engine.

Wraps three Hugging Face pipelines:
  * text   -> BERT multi-label toxicity classifier (6 categories)
  * image  -> ViT NSFW classifier
  * image  -> ViT AI-generated-image detector (informational signal)

Text and image results are combined with late fusion: each harm category keeps
its own score and content is flagged when any category reaches the threshold.
Models load lazily on first use, or eagerly with ``Moderator.load()``.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from PIL import Image

from . import config

logger = logging.getLogger(__name__)

Pipeline = Callable[..., Any]

AI_LABEL_WORDS = {"artificial", "ai", "fake", "generated", "synthetic"}
HUMAN_LABEL_WORDS = {"human", "real", "authentic", "natural"}
NSFW_CATEGORY = "NSFW image"


@dataclass
class ModerationResult:
    flagged: bool
    verdict: str
    top_category: Optional[str]
    top_score: float
    categories: Dict[str, float]
    modalities: List[str]
    threshold: float
    latency_ms: float
    ai_generated: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_list_of_results(raw: Any, n_inputs: int) -> List[List[Dict[str, Any]]]:
    """Normalise pipeline output to one list of {label, score} per input.

    Depending on the transformers version and input type, a pipeline returns
    either ``[{...}, ...]`` for a single input or ``[[{...}], ...]`` for many.
    """
    if isinstance(raw, dict):
        raw = [raw]
    if raw and isinstance(raw[0], dict):
        if n_inputs != 1:
            raise ValueError("Pipeline returned one result for several inputs")
        return [raw]
    return [list(r) if not isinstance(r, dict) else [r] for r in raw]


def _scores(items: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {str(i["label"]): float(i["score"]) for i in items}


def _label_words(label: str) -> set:
    return set(re.split(r"[^a-z0-9]+", label.lower())) - {""}


def ai_generated_probability(scores: Dict[str, float]) -> Optional[float]:
    """Pick the 'AI-generated' probability out of a detector's label scores."""
    for label, score in scores.items():
        if _label_words(label) & AI_LABEL_WORDS:
            return score
    for label, score in scores.items():
        if _label_words(label) & HUMAN_LABEL_WORDS:
            return 1.0 - score
    return None


def nsfw_probability(scores: Dict[str, float]) -> float:
    for label, score in scores.items():
        if "nsfw" in label.lower() or "porn" in label.lower() or "sexy" in label.lower():
            return score
    # Binary model whose positive label we don't recognise: fall back to 1 - normal.
    for label, score in scores.items():
        if label.lower() in {"normal", "safe", "sfw", "neutral"}:
            return 1.0 - score
    raise ValueError(f"Unrecognised NSFW model labels: {sorted(scores)}")


def to_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy; handles palette, alpha and animated (first frame) images."""
    if getattr(image, "is_animated", False):
        image.seek(0)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")


class Moderator:
    """Loads the models once and runs moderation requests against them."""

    def __init__(
        self,
        text_pipeline: Optional[Pipeline] = None,
        nsfw_pipeline: Optional[Pipeline] = None,
        ai_pipeline: Optional[Pipeline] = None,
        text_model: str = config.TEXT_MODEL,
        nsfw_model: str = config.NSFW_IMAGE_MODEL,
        ai_model: Optional[str] = config.AI_IMAGE_MODEL,
    ):
        self._pipes: Dict[str, Optional[Pipeline]] = {
            "text": text_pipeline,
            "nsfw": nsfw_pipeline,
            "ai": ai_pipeline,
        }
        self._names = {"text": text_model, "nsfw": nsfw_model, "ai": ai_model or None}
        self._tasks = {
            "text": "text-classification",
            "nsfw": "image-classification",
            "ai": "image-classification",
        }
        self._lock = threading.Lock()
        self._load_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------ loading
    def _build_pipeline(self, key: str) -> Pipeline:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s model %s", key, self._names[key])
        return pipeline(self._tasks[key], model=self._names[key], device=device)

    def _get(self, key: str) -> Optional[Pipeline]:
        if self._pipes[key] is not None:
            return self._pipes[key]
        if not self._names.get(key):
            return None
        with self._lock:
            if self._pipes[key] is None and key not in self._load_errors:
                try:
                    self._pipes[key] = self._build_pipeline(key)
                except Exception as exc:  # keep serving the other modalities
                    if key in ("text", "nsfw"):
                        raise
                    logger.warning("Optional model %s failed to load: %s", key, exc)
                    self._load_errors[key] = str(exc)
        return self._pipes[key]

    def load(self) -> "Moderator":
        """Load every configured model now (avoids a slow first request)."""
        for key in ("text", "nsfw", "ai"):
            self._get(key)
        return self

    def status(self) -> Dict[str, Any]:
        return {
            "models": {k: v for k, v in self._names.items() if v},
            "loaded": {k: self._pipes[k] is not None for k in self._pipes if self._names.get(k)},
            "errors": dict(self._load_errors),
        }

    # ---------------------------------------------------------------- scoring
    def text_scores(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        pipe = self._get("text")
        raw = pipe(
            list(texts),
            top_k=None,
            function_to_apply="sigmoid",  # multi-label: categories are independent
            truncation=True,
            max_length=512,
        )
        results = _as_list_of_results(raw, len(texts))
        return [
            {config.TEXT_LABEL_NAMES.get(k, k): v for k, v in _scores(r).items()}
            for r in results
        ]

    def image_scores(self, image: Image.Image) -> Dict[str, Any]:
        image = to_rgb(image)
        nsfw_raw = _as_list_of_results(self._get("nsfw")(image, top_k=None), 1)[0]
        out: Dict[str, Any] = {"categories": {NSFW_CATEGORY: nsfw_probability(_scores(nsfw_raw))}}
        ai_pipe = self._get("ai")
        out["ai_generated"] = None
        if ai_pipe is not None:
            ai_raw = _as_list_of_results(ai_pipe(image, top_k=None), 1)[0]
            out["ai_generated"] = ai_generated_probability(_scores(ai_raw))
        return out

    # ----------------------------------------------------------------- public
    def moderate(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        threshold: float = config.DEFAULT_THRESHOLD,
    ) -> ModerationResult:
        """Moderate text, an image, or both together."""
        text = (text or "").strip() or None
        if text is None and image is None:
            raise ValueError("Provide text, an image, or both.")
        _check_threshold(threshold)
        if text is not None and len(text) > config.MAX_TEXT_CHARS:
            raise ValueError(f"Text is longer than {config.MAX_TEXT_CHARS} characters.")

        start = time.perf_counter()
        categories: Dict[str, float] = {}
        modalities: List[str] = []
        notes: List[str] = []
        ai_generated = None

        if text is not None:
            categories.update(self.text_scores([text])[0])
            modalities.append("text")
        if image is not None:
            img = self.image_scores(image)
            categories.update(img["categories"])
            ai_generated = img["ai_generated"]
            modalities.append("image")
            if self._names.get("ai") and ai_generated is None:
                notes.append("AI-generated detector unavailable.")

        latency_ms = (time.perf_counter() - start) * 1000
        return _build_result(categories, modalities, threshold, latency_ms, ai_generated, notes)

    def moderate_batch(
        self, texts: Sequence[str], threshold: float = config.DEFAULT_THRESHOLD
    ) -> List[ModerationResult]:
        """Moderate many texts in one batched forward pass."""
        _check_threshold(threshold)
        if not texts:
            return []
        if len(texts) > config.MAX_BATCH_SIZE:
            raise ValueError(f"At most {config.MAX_BATCH_SIZE} texts per batch.")
        cleaned = [t.strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("Batch contains an empty text.")
        if any(len(t) > config.MAX_TEXT_CHARS for t in cleaned):
            raise ValueError(f"A text is longer than {config.MAX_TEXT_CHARS} characters.")

        start = time.perf_counter()
        all_scores = self.text_scores(cleaned)
        per_item_ms = (time.perf_counter() - start) * 1000 / len(cleaned)
        return [
            _build_result(s, ["text"], threshold, per_item_ms, None, []) for s in all_scores
        ]


def _check_threshold(threshold: float) -> None:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Threshold must be greater than 0 and at most 1.")


def _build_result(
    categories: Dict[str, float],
    modalities: List[str],
    threshold: float,
    latency_ms: float,
    ai_generated: Optional[float],
    notes: List[str],
) -> ModerationResult:
    top_category, top_score = max(categories.items(), key=lambda kv: kv[1])
    flagged = top_score >= threshold
    if flagged:
        verdict = f"Flagged: {top_category} ({top_score:.0%})"
    else:
        verdict = "Safe"
    return ModerationResult(
        flagged=flagged,
        verdict=verdict,
        top_category=top_category if flagged else None,
        top_score=round(top_score, 4),
        categories={k: round(v, 4) for k, v in categories.items()},
        modalities=modalities,
        threshold=threshold,
        latency_ms=round(latency_ms, 1),
        ai_generated=None if ai_generated is None else round(ai_generated, 4),
        notes=notes,
    )
