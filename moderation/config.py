"""Model names and limits. Every value can be overridden with an environment variable."""

import os

# Text: BERT (bert-base-uncased) fine-tuned on the Jigsaw Toxic Comment dataset.
TEXT_MODEL = os.getenv("TEXT_MODEL", "unitary/toxic-bert")

# Images: Vision Transformer fine-tuned to separate normal from NSFW images.
NSFW_IMAGE_MODEL = os.getenv("NSFW_IMAGE_MODEL", "Falconsai/nsfw_image_detection")

# Images: Vision Transformer that estimates whether an image is AI-generated.
# This is reported as a separate signal and does not flag content on its own.
# Set AI_IMAGE_MODEL="" to turn it off.
AI_IMAGE_MODEL = os.getenv("AI_IMAGE_MODEL", "umm-maybe/AI-image-detector")

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "5000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

# Readable names for the toxic-bert labels.
TEXT_LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxicity",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}
