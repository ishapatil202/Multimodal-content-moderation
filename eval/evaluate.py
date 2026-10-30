"""
Measure real accuracy and latency of the deployed models.

    pip install -r requirements-dev.txt
    python eval/evaluate.py                 # 2,000 balanced comments + latency
    python eval/evaluate.py --n 500         # quicker run

Text accuracy uses the public Civil Comments test set (comments labelled toxic
when >= 50% of annotators said so). toxic-bert was trained on the older Jigsaw
2018 data, so this is an out-of-distribution check rather than its own test set.
The sample is balanced (half toxic, half non-toxic) and the seed is fixed so
results are reproducible. The decision threshold is tuned for best F1 on the
validation split, then test results are reported at the default threshold and
at the tuned one (use --no-tune to skip, --skip-latency to rerun faster).

Image latency is measured on random images; there is no image accuracy test
because there is no openly licensed labelled NSFW benchmark to ship here.

Results are written to eval/results.json and eval/results.md.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation import config  # noqa: E402
from moderation.engine import Moderator  # noqa: E402

OUT = Path(__file__).resolve().parent


def load_civil_comments(n, seed, split="test"):
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split=split)
    toxic = ds.filter(lambda r: r["toxicity"] >= 0.5 and len(r["text"].strip()) > 0)
    clean = ds.filter(lambda r: r["toxicity"] < 0.5 and len(r["text"].strip()) > 0)
    half = n // 2
    toxic = toxic.shuffle(seed=seed).select(range(min(half, len(toxic))))
    clean = clean.shuffle(seed=seed).select(range(min(half, len(clean))))
    texts = [t[: config.MAX_TEXT_CHARS] for t in list(toxic["text"]) + list(clean["text"])]
    labels = [1] * len(toxic) + [0] * len(clean)
    return texts, labels


def percentile(values, p):
    return float(np.percentile(np.array(values), p))


def score_texts(mod, texts, batch_size):
    scores = []
    for i in range(0, len(texts), batch_size):
        for s in mod.text_scores(texts[i : i + batch_size]):
            scores.append(max(s.values()))  # same rule the app uses
    return scores


def metrics(labels, scores, threshold):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    preds = [int(s >= threshold) for s in scores]
    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def tune_threshold(mod, n, seed, batch_size):
    """Pick the threshold with the best F1 on the validation split (never on test)."""
    texts, labels = load_civil_comments(n, seed, split="validation")
    scores = score_texts(mod, texts, batch_size)
    grid = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    best = max((metrics(labels, scores, t) for t in grid), key=lambda m: m["f1"])
    return best["threshold"], best


def evaluate_text(mod, n, seed, thresholds, batch_size):
    from sklearn.metrics import roc_auc_score

    texts, labels = load_civil_comments(n, seed)
    scores = score_texts(mod, texts, batch_size)
    return {
        "dataset": "google/civil_comments (test), balanced sample",
        "n": len(texts),
        "roc_auc": roc_auc_score(labels, scores),
        "rows": [metrics(labels, scores, t) for t in thresholds],
    }


def time_calls(fn, inputs, warmup=3):
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs:
        t = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t) * 1000)
    return {
        "n": len(times),
        "p50_ms": percentile(times, 50),
        "p95_ms": percentile(times, 95),
        "mean_ms": statistics.mean(times),
    }


def evaluate_latency(mod, n, seed):
    rng = np.random.default_rng(seed)
    sentences = [
        "Thanks for the update, see you tomorrow.",
        "This is the dumbest thing I have read all week.",
        "Can someone explain how the new policy affects renters in my city?",
        "Great photo! Where was this taken?",
    ]
    texts = [sentences[i % len(sentences)] for i in range(n)]
    images = [
        Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)) for _ in range(min(n, 50))
    ]
    return {
        "text_single_request": time_calls(lambda t: mod.moderate(text=t), texts),
        "image_single_request": time_calls(lambda im: mod.moderate(image=im), images),
        "text_plus_image": time_calls(
            lambda im: mod.moderate(text=sentences[0], image=im), images[:20]
        ),
    }


def hardware():
    import torch

    info = {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count()}
    info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    info["torch_threads"] = torch.get_num_threads()
    return info


def to_markdown(res):
    lines = [
        "# Evaluation results",
        "",
        f"Hardware: {res['hardware']['device']}, {res['hardware']['cpus']} CPUs, "
        f"Python {res['hardware']['python']}",
    ]
    if "text" in res:
        t = res["text"]
        lines += ["", f"## Text ({t['dataset']}, n={t['n']}, ROC-AUC={t['roc_auc']:.3f})", ""]
        if "tuning" in res:
            tu = res["tuning"]
            lines += [
                f"Threshold {tu['threshold']} was chosen for best F1 on a separate "
                f"validation sample (n={tu['n']}); the test set was not used for tuning.",
                "",
            ]
        lines += [
            "| Threshold | Accuracy | Precision | Recall | F1 | Macro-F1 |",
            "|---|---|---|---|---|---|",
        ]
        for r in t["rows"]:
            lines.append(
                f"| {r['threshold']} | {r['accuracy']:.1%} | {r['precision']:.1%} | "
                f"{r['recall']:.1%} | {r['f1']:.1%} | {r['macro_f1']:.1%} |"
            )
    if "latency" in res:
        lines += [
            "",
            "## Latency per request (model inference, excludes network)",
            "",
            "| Request | n | p50 | p95 |",
            "|---|---|---|---|",
        ]
        for name, v in res["latency"].items():
            lines.append(
                f"| {name.replace('_', ' ')} | {v['n']} | {v['p50_ms']:.0f} ms | {v['p95_ms']:.0f} ms |"
            )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=2000, help="texts in the accuracy sample")
    p.add_argument("--latency-n", type=int, default=100)
    p.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tune", action="store_true", help="skip threshold tuning on validation")
    p.add_argument("--skip-accuracy", action="store_true")
    p.add_argument("--skip-latency", action="store_true")
    args = p.parse_args()

    mod = Moderator().load()
    res = {"hardware": hardware(), "models": mod.status()["models"]}
    if not args.skip_latency:
        res["latency"] = evaluate_latency(mod, args.latency_n, args.seed)
    if not args.skip_accuracy:
        thresholds = [args.threshold]
        if not args.no_tune:
            best, val = tune_threshold(mod, args.n, args.seed, args.batch_size)
            res["tuning"] = {"threshold": best, "n": args.n, "validation": val}
            if best != args.threshold:
                thresholds.append(best)
        res["text"] = evaluate_text(mod, args.n, args.seed, thresholds, args.batch_size)

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    (OUT / "results.md").write_text(md)
    print(md)
    # Skip interpreter teardown: avoids a harmless but noisy multiprocess
    # error on Python 3.12.0 after the results are already saved.
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
