"""
Entry point for Hugging Face Spaces and local runs.

    python app.py            # UI at http://localhost:7860, API docs at /docs

The Gradio UI is mounted on the FastAPI app, so one server provides both.
"""

import logging
import os

import gradio as gr
import uvicorn

from moderation.api import create_api
from moderation.engine import Moderator
from moderation.ui import build_ui

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(moderator: Moderator):
    api = create_api(moderator)
    return gr.mount_gradio_app(api, build_ui(moderator), path="/")


moderator = Moderator()
app = create_app(moderator)

if __name__ == "__main__":
    moderator.load()  # download/load all models before accepting traffic
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
