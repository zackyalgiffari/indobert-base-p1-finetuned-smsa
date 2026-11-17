"""Load the IndoNLU SmSA sentiment dataset as a Hugging Face DatasetDict.

The Hub copy (`indonlp/indonlu`) is a loading-script dataset, which `datasets>=4`
no longer executes, so the raw TSVs are read straight from the IndoNLU GitHub repo,
pinned to a commit for reproducibility.
"""

from __future__ import annotations

import csv

import pandas as pd
from datasets import Dataset, DatasetDict

INDONLU_COMMIT = "ce728f6926a36174b9923dfe49d6a6839b6e9bb7"
BASE_URL = (
    "https://raw.githubusercontent.com/IndoNLP/indonlu/"
    f"{INDONLU_COMMIT}/dataset/smsa_doc-sentiment-prosa"
)
SPLIT_FILES = {
    "train": "train_preprocess.tsv",
    "validation": "valid_preprocess.tsv",
    "test": "test_preprocess.tsv",
}

LABELS = ["positive", "neutral", "negative"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def _read_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{BASE_URL}/{SPLIT_FILES[name]}",
        sep="\t",
        header=None,
        names=["text", "label"],
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
    )
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"{name}: unexpected labels {sorted(unknown)}")
    df["label"] = df["label"].map(LABEL2ID)
    return df


def load_smsa() -> DatasetDict:
    return DatasetDict(
        {name: Dataset.from_pandas(_read_split(name), preserve_index=False) for name in SPLIT_FILES}
    )


if __name__ == "__main__":
    ds = load_smsa()
    print(ds)
    for split in ds:
        counts = pd.Series(ds[split]["label"]).map(ID2LABEL).value_counts().to_dict()
        print(split, counts)
