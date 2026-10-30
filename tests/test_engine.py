import pytest
from PIL import Image

from moderation import config
from moderation.engine import (
    _as_list_of_results,
    ai_generated_probability,
    nsfw_probability,
    to_rgb,
)
from tests.fakes import make_moderator


@pytest.fixture
def mod():
    return make_moderator()


def red():
    return Image.new("RGB", (32, 32), (255, 0, 0))


def blue():
    return Image.new("RGB", (32, 32), (0, 0, 255))


# ------------------------------------------------------------------- text
def test_clean_text_is_safe(mod):
    r = mod.moderate(text="Have a lovely day")
    assert not r.flagged and r.verdict == "Safe" and r.top_category is None
    assert r.modalities == ["text"]
    assert set(r.categories) == set(config.TEXT_LABEL_NAMES.values())


def test_single_toxic_word_is_flagged(mod):
    # Regression: the old keyword demo returned "safe" for one bad word at threshold 0.5.
    r = mod.moderate(text="you idiot")
    assert r.flagged and r.top_category == "Toxic"
    assert r.verdict.startswith("Flagged: Toxic")


def test_threat_category(mod):
    r = mod.moderate(text="I will hurt you")
    assert r.flagged and r.categories["Threat"] > 0.8


def test_lower_threshold_is_stricter(mod):
    text = "you idiot"
    assert mod.moderate(text=text, threshold=0.5).flagged
    assert not mod.moderate(text=text, threshold=0.99).flagged
    assert mod.moderate(text="nice", threshold=0.01).flagged  # 0.01 baseline scores reach it


def test_text_pipeline_called_as_multilabel(mod):
    mod.moderate(text="hello")
    _, kwargs = mod._pipes["text"].calls[-1]
    assert kwargs["function_to_apply"] == "sigmoid" and kwargs["top_k"] is None
    assert kwargs["truncation"] is True


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_requires_input(mod, bad):
    with pytest.raises(ValueError):
        mod.moderate(text=bad)


@pytest.mark.parametrize("th", [0, -0.1, 1.5])
def test_bad_threshold(mod, th):
    with pytest.raises(ValueError):
        mod.moderate(text="hi", threshold=th)


def test_text_too_long(mod):
    with pytest.raises(ValueError):
        mod.moderate(text="a" * (config.MAX_TEXT_CHARS + 1))


def test_batch(mod):
    results = mod.moderate_batch(["hi", "you idiot", "I will hurt you"])
    assert [r.flagged for r in results] == [False, True, True]
    assert len(mod._pipes["text"].calls) == 1  # one batched call


def test_batch_validation(mod):
    assert mod.moderate_batch([]) == []
    with pytest.raises(ValueError):
        mod.moderate_batch(["ok", " "])
    with pytest.raises(ValueError):
        mod.moderate_batch(["x"] * (config.MAX_BATCH_SIZE + 1))


# ------------------------------------------------------------------ image
def test_safe_image(mod):
    r = mod.moderate(image=blue())
    assert not r.flagged and r.modalities == ["image"]
    assert r.ai_generated == pytest.approx(0.03)


def test_unsafe_image(mod):
    r = mod.moderate(image=red())
    assert r.flagged and r.top_category == "NSFW image"
    assert r.ai_generated == pytest.approx(0.03)


def test_ai_signal_does_not_flag_by_itself(mod):
    # AI-generated score is informational; only harm categories flag.
    r = mod.moderate(image=Image.new("RGB", (8, 8), (0, 255, 0)))
    assert r.ai_generated == pytest.approx(0.95)
    assert not r.flagged
    assert set(r.categories) == {"NSFW image"}


def test_rgba_and_palette_images_converted(mod):
    rgba = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
    pal = blue().convert("P")
    mod.moderate(image=rgba)
    mod.moderate(image=pal)
    assert set(mod._pipes["nsfw"].modes) == {"RGB"}


def test_transparent_pixels_become_white():
    img = to_rgb(Image.new("RGBA", (2, 2), (0, 0, 0, 0)))
    assert img.mode == "RGB" and img.getpixel((0, 0)) == (255, 255, 255)


def test_without_ai_model():
    m = make_moderator(ai=False)
    r = m.moderate(image=blue())
    assert r.ai_generated is None and r.notes == []


# -------------------------------------------------------------- multimodal
def test_multimodal_late_fusion(mod):
    r = mod.moderate(text="lovely photo", image=red())
    assert r.modalities == ["text", "image"]
    assert r.flagged and r.top_category == "NSFW image"
    r2 = mod.moderate(text="you idiot", image=blue())
    assert r2.flagged and r2.top_category == "Toxic"
    r3 = mod.moderate(text="lovely photo", image=blue())
    assert not r3.flagged


# ----------------------------------------------------------------- helpers
def test_output_shapes_normalised():
    one = [{"label": "a", "score": 0.1}]
    assert _as_list_of_results(one, 1) == [one]
    assert _as_list_of_results([one, one], 2) == [one, one]
    assert _as_list_of_results({"label": "a", "score": 0.1}, 1) == [one]


@pytest.mark.parametrize(
    "scores,expected",
    [
        ({"artificial": 0.8, "human": 0.2}, 0.8),
        ({"AI": 0.7, "Real": 0.3}, 0.7),
        ({"human": 0.9}, pytest.approx(0.1)),
        ({"cat": 0.5}, None),
    ],
)
def test_ai_label_parsing(scores, expected):
    assert ai_generated_probability(scores) == expected


def test_ai_label_parsing_ignores_substrings():
    # "ai" inside another word must not count as the AI label.
    assert ai_generated_probability({"maintained": 0.9, "human": 0.4}) == pytest.approx(0.6)


def test_nsfw_label_parsing():
    assert nsfw_probability({"normal": 0.2, "nsfw": 0.8}) == 0.8
    assert nsfw_probability({"normal": 0.25}) == 0.75
    with pytest.raises(ValueError):
        nsfw_probability({"cat": 1.0})
