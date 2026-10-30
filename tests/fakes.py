"""Test doubles that mimic Hugging Face pipeline outputs, so tests run without model downloads."""

TEXT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _text_scores(text):
    t = text.lower()
    scores = dict.fromkeys(TEXT_LABELS, 0.01)
    if "idiot" in t or "hate" in t:
        scores.update(toxic=0.97, insult=0.91)
    if "hurt you" in t:
        scores.update(toxic=0.88, threat=0.83)
    return [{"label": k, "score": v} for k, v in scores.items()]


class FakeTextPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        if isinstance(inputs, str):
            return _text_scores(inputs)
        return [_text_scores(t) for t in inputs]


class FakeImagePipeline:
    """Scores by colour: images whose first pixel equals `trigger` count as 'positive'."""

    def __init__(self, positive, negative, trigger=(255, 0, 0)):
        self.positive, self.negative, self.trigger = positive, negative, trigger
        self.modes = []

    def __call__(self, image, **kwargs):
        self.modes.append(image.mode)
        p = 0.95 if image.getpixel((0, 0))[:3] == self.trigger else 0.03
        return [{"label": self.positive, "score": p}, {"label": self.negative, "score": 1 - p}]


def make_moderator(ai=True):
    from moderation.engine import Moderator

    return Moderator(
        text_pipeline=FakeTextPipeline(),
        nsfw_pipeline=FakeImagePipeline("nsfw", "normal"),
        ai_pipeline=FakeImagePipeline("artificial", "human", trigger=(0, 255, 0)) if ai else None,
        ai_model="fake/ai" if ai else None,
    )
