---
language:
- id
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: indobenchmark/indobert-base-p1
tags:
- sentiment-analysis
- indonesian
- indobert
- indonlu
metrics:
- f1
- accuracy
---

# IndoBERT base — Indonesian sentiment (SmSA)

[`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1)
fine-tuned for 3-class Indonesian sentiment classification (`positive`, `neutral`,
`negative`) on **SmSA** from the [IndoNLU](https://github.com/IndoNLP/indonlu) benchmark.

Training code: https://github.com/zackyalgiffari/sentimen-model

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="zackyalgiffari/indobert-base-p1-smsa-sentiment")
clf("makanannya enak dan pelayanannya ramah")
```

## Evaluation

See `metrics.json` and `classification_report.txt` in this repository (SmSA validation
and test splits). The main metric is macro-F1, because `neutral` is only ~10% of the
training data.

## Training

- Data: SmSA train split (11,000 texts), model selection on the validation split
  (1,260), final report on the labelled test split (500).
- Hyperparameters: learning rate 2e-5, batch size 32, 5 epochs, linear schedule with
  10% warmup, weight decay 0.01, max length 256, bf16, seed 42.
- Hardware: Google Colab GPU (NVIDIA L4).

## Limitations

SmSA consists mostly of reviews and comments in formal-to-casual Indonesian. Expect
lower accuracy on heavy slang, code-mixing, regional languages, sarcasm, or domains far
from reviews. The `neutral` class is the hardest and least represented.

## Citation

```bibtex
@inproceedings{wilie2020indonlu,
  title     = {IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author    = {Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and Xiaohong Li and Zhi Yuan Lim and Sidik Soleman and Rahmad Mahendra and Pascale Fung and Syafri Bahar and Ayu Purwarianti},
  booktitle = {Proceedings of AACL-IJCNLP 2020},
  year      = {2020}
}
```
