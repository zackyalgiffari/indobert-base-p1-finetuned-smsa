# indobert-base-p1-finetuned-smsa

Fine-tuning **IndoBERT** (`indobenchmark/indobert-base-p1`) for Indonesian sentiment
analysis (positive / neutral / negative) on the public **SmSA** dataset from the
[IndoNLU](https://github.com/IndoNLP/indonlu) benchmark. Training runs on a Google
Colab GPU through the [`colab` CLI](https://pypi.org/project/google-colab-cli/).

## Results

<!-- RESULTS:START -->
Model: [`zackyalgiffari/indobert-base-p1-finetuned-smsa`](https://huggingface.co/zackyalgiffari/indobert-base-p1-finetuned-smsa)
— trained on a Colab **NVIDIA L4** (bf16) in 2.5 minutes,
transformers 5.22.1, torch 2.14.1+cu130.

| Split | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---:|---:|---:|---:|
| validation (1,260) | 94.29 | 92.40 | 91.83 | **92.04** |
| test (500) | 92.00 | 92.50 | 88.12 | **89.74** |

Per class (test): positive F1 93.78 · neutral F1 81.29 · negative F1 94.15.
`neutral` is the hardest class: recall is 71.6% (25 of 88 neutral texts are predicted
as positive or negative), while precision stays high (94.0%).

<details><summary>Validation per epoch (best model = epoch 3)</summary>

| Epoch | Accuracy | Macro-F1 | Loss |
|---:|---:|---:|---:|
| 1 | 93.10 | 90.44 | 0.1863 |
| 2 | 94.13 | 91.32 | 0.1785 |
| 3 | 94.29 | 92.04 | 0.2227 |
| 4 | 94.05 | 91.42 | 0.2695 |
| 5 | 94.29 | 92.01 | 0.2874 |

Validation loss rises after epoch 2–3 while F1 plateaus — the model starts to overfit;
`load_best_model_at_end` picks the checkpoint by macro-F1.
</details>

Details: [`results/metrics.json`](results/metrics.json),
[`results/classification_report.txt`](results/classification_report.txt),
confusion matrix [validation](results/confusion_matrix_validation.png) /
[test](results/confusion_matrix_test.png).
<!-- RESULTS:END -->

## Dataset

SmSA (*Sentence-level / document-level Sentiment Analysis*, Prosa.ai) — Indonesian
reviews and comments, 3 classes.

| Split      |  Total | positive | neutral | negative |
|------------|-------:|---------:|--------:|---------:|
| train      | 11,000 |    6,416 |   1,148 |    3,436 |
| validation |  1,260 |      735 |     131 |      394 |
| test       |    500 |      208 |      88 |      204 |

`neutral` is only ~10% of the training data, so the headline metric is **macro-F1**
(the unweighted mean of per-class F1), not accuracy.

The Hugging Face Hub copy (`indonlp/indonlu`) is a loading-script dataset, which
`datasets>=4` no longer runs, so `src/data.py` reads the raw TSVs straight from the
IndoNLU GitHub repo, pinned to commit
[`ce728f6`](https://github.com/IndoNLP/indonlu/tree/ce728f6926a36174b9923dfe49d6a6839b6e9bb7/dataset/smsa_doc-sentiment-prosa).

## Layout

```
src/data.py            download & parse SmSA → DatasetDict train/validation/test
src/train.py           fine-tuning (HF Trainer), evaluation, push to the Hub
src/predict.py         command-line inference
scripts/colab_train.sh local launcher: create Colab VM → upload → train → fetch results → stop VM
scripts/remote_job.sh  runs on the Colab VM (install deps + train)
MODEL_CARD.md          model card uploaded to the Hugging Face Hub with the weights
results/               metrics, classification report, confusion matrices
```

## Running

Prerequisites: the `colab` CLI is logged in (`colab usage` works) and, to push to the
Hub, a Hugging Face token at `~/.cache/huggingface/token`.

```bash
# quick pipeline check (small subset, 1 epoch)
scripts/colab_train.sh --smoke

# full training + upload the model to the Hugging Face Hub (private repo)
scripts/colab_train.sh --push-to-hub zackyalgiffari/indobert-base-p1-finetuned-smsa

# another GPU / other hyperparameters
GPU=A100 scripts/colab_train.sh --epochs 3 --lr 3e-5
```

Every argument is forwarded to `python -m src.train` (see `python -m src.train --help`).
The Colab VM is always stopped when the script exits, including on failure.

Default hyperparameters: lr `2e-5`, batch 32, 5 epochs, 10% warmup, weight decay 0.01,
`max_length` 256, bf16 (when the GPU supports it), seed 42. The best checkpoint is
selected by validation macro-F1.

## Inference

```bash
pip install -r requirements.txt torch
python -m src.predict "makanannya enak dan pelayanannya ramah" "pengirimannya lambat sekali"
```

```python
from transformers import pipeline
clf = pipeline("text-classification", model="zackyalgiffari/indobert-base-p1-finetuned-smsa")
clf("tempatnya nyaman, pasti balik lagi")
```

## License & attribution

Code: MIT. Base model: [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1) (MIT).
Dataset: SmSA from IndoNLU (the IndoNLU repo is Apache-2.0) — please cite:

```bibtex
@inproceedings{wilie2020indonlu,
  title     = {IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author    = {Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and Xiaohong Li and Zhi Yuan Lim and Sidik Soleman and Rahmad Mahendra and Pascale Fung and Syafri Bahar and Ayu Purwarianti},
  booktitle = {Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year      = {2020}
}
```
