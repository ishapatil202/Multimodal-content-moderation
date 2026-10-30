"""REST API (FastAPI). Interactive docs are served at /docs."""

from __future__ import annotations

import io
from typing import List, Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import config
from .engine import Moderator


class TextRequest(BaseModel):
    text: str = Field(..., description="Text to moderate", examples=["Have a great day!"])
    threshold: float = Field(config.DEFAULT_THRESHOLD, gt=0, le=1)


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to moderate")
    threshold: float = Field(config.DEFAULT_THRESHOLD, gt=0, le=1)


class ModerationResponse(BaseModel):
    flagged: bool
    verdict: str
    top_category: Optional[str]
    top_score: float
    categories: dict
    modalities: List[str]
    threshold: float
    latency_ms: float
    ai_generated: Optional[float] = None
    notes: List[str] = []


class BatchResponse(BaseModel):
    results: List[ModerationResponse]
    count: int


def _read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(config.MAX_IMAGE_BYTES + 1)
    if len(data) > config.MAX_IMAGE_BYTES:
        raise HTTPException(413, f"Image is larger than {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    if not data:
        raise HTTPException(400, "Uploaded image is empty.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "File is not a readable image.") from exc


def build_router(moderator: Moderator) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["moderation"])

    def run(**kwargs):
        try:
            return moderator.moderate(**kwargs).to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    # Endpoints are plain `def` so FastAPI runs model inference in a worker thread.
    @router.get("/health")
    def health():
        return {"status": "ok", **moderator.status()}

    @router.post("/moderate/text", response_model=ModerationResponse)
    def moderate_text(req: TextRequest):
        return run(text=req.text, threshold=req.threshold)

    @router.post("/moderate/batch", response_model=BatchResponse)
    def moderate_batch(req: BatchRequest):
        try:
            results = moderator.moderate_batch(req.texts, req.threshold)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"results": [r.to_dict() for r in results], "count": len(results)}

    @router.post("/moderate/image", response_model=ModerationResponse)
    def moderate_image(
        image: UploadFile = File(...),
        threshold: float = Form(config.DEFAULT_THRESHOLD, gt=0, le=1),
    ):
        return run(image=_read_image(image), threshold=threshold)

    @router.post("/moderate", response_model=ModerationResponse)
    def moderate_multimodal(
        text: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
        threshold: float = Form(config.DEFAULT_THRESHOLD, gt=0, le=1),
    ):
        pil = _read_image(image) if image is not None and image.filename else None
        return run(text=text, image=pil, threshold=threshold)

    return router


def create_api(moderator: Moderator) -> FastAPI:
    app = FastAPI(
        title="Multimodal Content Moderation API",
        description="BERT text toxicity + ViT image moderation. UI at `/`, REST under `/api`.",
        version="2.0.0",
    )
    app.include_router(build_router(moderator))
    return app
