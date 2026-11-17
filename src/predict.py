"""Classify Indonesian text with a fine-tuned sentiment model.

    python -m src.predict "makanannya enak banget" "pelayanannya lambat"
    python -m src.predict --model outputs/final "..."
"""

from __future__ import annotations

import argparse

from transformers import pipeline

DEFAULT_MODEL = "zackyalgiffari/indobert-base-p1-smsa-sentiment"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("texts", nargs="+")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hub repo id or local directory")
    args = p.parse_args()

    clf = pipeline("text-classification", model=args.model, top_k=None)
    for text, scores in zip(args.texts, clf(args.texts)):
        best = max(scores, key=lambda s: s["score"])
        detail = ", ".join(f"{s['label']}={s['score']:.3f}" for s in scores)
        print(f"{best['label']:>8}  {text}\n          ({detail})")


if __name__ == "__main__":
    main()
