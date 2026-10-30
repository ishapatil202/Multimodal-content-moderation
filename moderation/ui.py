"""Gradio web UI."""

from __future__ import annotations

import gradio as gr

from . import config
from .engine import Moderator

TEXT_EXAMPLES = [
    "Thanks for sharing this, really helpful write-up!",
    "You are a worthless idiot and everyone here hates you.",
    "I'm going to find where you live and hurt you.",
    "This movie was so bad it was hilarious.",
]

INTRO = """
# 🛡️ Multimodal Content Moderation
Checks **text** with a BERT toxicity classifier and **images** with Vision Transformer
classifiers (NSFW + AI-generated). Content is flagged when any category's score reaches
the threshold: **lower threshold = stricter**. REST API docs: [`/docs`](/docs).
"""


def _render(result):
    icon = "🚩" if result.flagged else "✅"
    lines = [f"## {icon} {result.verdict}"]
    lines.append(
        f"Modalities: {', '.join(result.modalities)} · threshold {result.threshold:.2f}"
        f" · {result.latency_ms:.0f} ms"
    )
    if result.ai_generated is not None:
        lines.append(
            f"\n**AI-generated image likelihood:** {result.ai_generated:.0%}"
            " _(informational, does not flag content)_"
        )
    for note in result.notes:
        lines.append(f"\n_{note}_")
    return "\n".join(lines), result.categories, result.to_dict()


def build_ui(moderator: Moderator) -> gr.Blocks:
    def run(text=None, image=None, threshold=config.DEFAULT_THRESHOLD):
        try:
            return _render(moderator.moderate(text=text, image=image, threshold=threshold))
        except ValueError as exc:
            raise gr.Error(str(exc))

    def outputs():
        verdict = gr.Markdown()
        scores = gr.Label(label="Category scores", num_top_classes=7, show_heading=False)
        with gr.Accordion("Raw JSON", open=False):
            raw = gr.JSON()
        return [verdict, scores, raw]

    with gr.Blocks(title="Content Moderation") as demo:
        gr.Markdown(INTRO)
        threshold = gr.Slider(0.05, 0.95, value=config.DEFAULT_THRESHOLD, step=0.05, label="Threshold")

        with gr.Tab("Text"):
            text_in = gr.Textbox(label="Text", lines=5, placeholder="Paste a comment or post…")
            text_btn = gr.Button("Analyze text", variant="primary")
            gr.Examples(TEXT_EXAMPLES, inputs=[text_in])
            text_out = outputs()
            text_btn.click(lambda t, th: run(text=t, threshold=th), [text_in, threshold], text_out)

        with gr.Tab("Image"):
            img_in = gr.Image(type="pil", label="Image")
            img_btn = gr.Button("Analyze image", variant="primary")
            img_out = outputs()
            img_btn.click(lambda i, th: run(image=i, threshold=th), [img_in, threshold], img_out)

        with gr.Tab("Text + Image"):
            with gr.Row():
                mm_text = gr.Textbox(label="Caption / post text", lines=5)
                mm_img = gr.Image(type="pil", label="Image")
            mm_btn = gr.Button("Analyze both", variant="primary")
            mm_out = outputs()
            mm_btn.click(
                lambda t, i, th: run(text=t, image=i, threshold=th),
                [mm_text, mm_img, threshold],
                mm_out,
            )

        gr.Markdown(
            "Models: "
            f"[{config.TEXT_MODEL}](https://huggingface.co/{config.TEXT_MODEL}) · "
            f"[{config.NSFW_IMAGE_MODEL}](https://huggingface.co/{config.NSFW_IMAGE_MODEL})"
            + (
                f" · [{config.AI_IMAGE_MODEL}](https://huggingface.co/{config.AI_IMAGE_MODEL})"
                if config.AI_IMAGE_MODEL
                else ""
            )
        )
    return demo
