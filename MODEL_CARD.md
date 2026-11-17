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
model-index:
- name: indobert-base-p1-finetuned-smsa
  results:
  - task:
      type: text-classification
      name: Sentiment Analysis
    dataset:
      name: IndoNLU SmSA
      type: indonlp/indonlu
      config: smsa
      split: test
    metrics:
    - type: f1
      name: Macro F1
      value: 0.8974
    - type: accuracy
      name: Accuracy
      value: 0.9200
---

# indobert-base-p1-finetuned-smsa

[`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1)
fine-tuned for 3-class Indonesian sentiment classification (`positive`, `neutral`,
`negative`) on **SmSA** from the [IndoNLU](https://github.com/IndoNLP/indonlu) benchmark.

Training code: https://github.com/zackyalgiffari/indobert-sentiment-indonesian

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="zackyalgiffari/indobert-base-p1-finetuned-smsa")
clf("makanannya enak dan pelayanannya ramah")
```

## Evaluation

SmSA, best checkpoint selected on validation macro-F1 (epoch 3).

| Split | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|
| validation (1,260) | 94.29 | 92.40 | 91.83 | **92.04** |
| test (500) | 92.00 | 92.50 | 88.12 | **89.74** |

Per-class test F1: positive 93.78, neutral 81.29, negative 94.15. `neutral` is the
hardest class (recall 71.6%) and is only ~10% of the training data, which is why
macro-F1 is the headline metric. Full numbers: `metrics.json`, `classification_report.txt`.

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
