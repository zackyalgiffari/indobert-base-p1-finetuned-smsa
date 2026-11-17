"""Fine-tune an Indonesian encoder for 3-way sentiment classification on SmSA."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import ID2LABEL, INDONLU_COMMIT, LABEL2ID, LABELS, load_smsa


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="indobenchmark/indobert-base-p1")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=float, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, to test the pipeline")
    p.add_argument("--push-to-hub", metavar="REPO_ID", help="upload the best model (private repo)")
    return p.parse_args()


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }


def save_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    print(f"[env] torch={torch.__version__} transformers={transformers.__version__} "
          f"cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'} "
          f"bf16={use_bf16} fp16={use_fp16}", flush=True)

    ds = load_smsa()
    if args.smoke:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(256))
        ds["validation"] = ds["validation"].select(range(128))
        ds["test"] = ds["test"].select(range(128))
        args.epochs = 1
    print(ds, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
        batched=True,
        remove_columns=["text"],
    )
    lengths = np.array([len(x) for x in tokenized["train"]["input_ids"]])
    truncated = int((lengths >= args.max_length).sum())
    print(f"[tokens] train p50={np.percentile(lengths, 50):.0f} p95={np.percentile(lengths, 95):.0f} "
          f"p99={np.percentile(lengths, 99):.0f} max={lengths.max()} truncated={truncated}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=2,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    start = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - start

    metrics: dict = {
        "model": args.model,
        "dataset": f"IndoNLU SmSA @ IndoNLP/indonlu@{INDONLU_COMMIT}",
        "smoke": args.smoke,
        "hyperparameters": {
            k: getattr(args, k)
            for k in ("max_length", "epochs", "lr", "batch_size", "warmup_ratio", "weight_decay", "seed")
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "precision": "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "train": {
            "seconds": round(train_seconds, 1),
            "loss": train_result.training_loss,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_f1_macro": trainer.state.best_metric,
        },
        "epoch_log": [
            {k: v for k, v in entry.items() if k.startswith("eval_") or k == "epoch"}
            for entry in trainer.state.log_history
            if "eval_f1_macro" in entry
        ],
    }

    report_lines = []
    for split in ("validation", "test"):
        out = trainer.predict(tokenized[split])
        preds = out.predictions.argmax(-1)
        labels = out.label_ids
        metrics[split] = {k.removeprefix("test_"): v for k, v in out.metrics.items()
                          if not k.endswith(("_runtime", "_per_second"))}
        report = classification_report(labels, preds, labels=list(range(len(LABELS))),
                                       target_names=LABELS, digits=4, zero_division=0)
        report_lines.append(f"=== {split} ===\n{report}")
        cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
        metrics[split]["confusion_matrix"] = cm.tolist()
        save_confusion_matrix(cm, results_dir / f"confusion_matrix_{split}.png", f"SmSA {split}")
        print(f"[{split}] {json.dumps(metrics[split])}", flush=True)

    (results_dir / "classification_report.txt").write_text("\n".join(report_lines))
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print("\n".join(report_lines), flush=True)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if args.push_to_hub:
        from huggingface_hub import HfApi

        token_path = os.environ.get("HF_TOKEN_PATH")
        token = Path(token_path).read_text().strip() if token_path else None
        api = HfApi(token=token)
        api.create_repo(args.push_to_hub, repo_type="model", private=True, exist_ok=True)
        model_card = Path("MODEL_CARD.md")
        if model_card.exists():
            (final_dir / "README.md").write_text(model_card.read_text())
        for name in ("metrics.json", "classification_report.txt"):
            (final_dir / name).write_text((results_dir / name).read_text())
        api.upload_folder(
            repo_id=args.push_to_hub,
            folder_path=str(final_dir),
            commit_message=f"Fine-tuned {args.model} on SmSA "
                           f"(test f1_macro={metrics['test']['f1_macro']:.4f})",
        )
        print(f"[hub] pushed to https://huggingface.co/{args.push_to_hub}", flush=True)


if __name__ == "__main__":
    main()
