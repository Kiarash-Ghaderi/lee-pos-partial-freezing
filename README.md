# Partial Freezing of a Multilingual Distilled Model for Persian POS Tagging

**Researcher:** Kiarash Ghaderi

## Overview

This project investigates the impact of partial Transformer-layer freezing on multilingual DistilBERT fine-tuning for Persian Part-of-Speech (POS) tagging.

Two training strategies are compared:

- **Full Fine-Tuning:** All model parameters remain trainable.
- **Partial Freezing:** The lower three Transformer blocks are frozen, while the embeddings, upper three Transformer blocks, and classification head remain trainable.

The objective is to evaluate the trade-off between predictive performance and computational efficiency.

## Dataset

The experiments use the Universal Dependencies Persian PerDT treebank.

- Training subset: 12,000 sentences, sampled with random seed 42
- Validation set: 1,456 sentences
- Test set: 1,455 sentences
- Observed UPOS classes: 16

The UPOS class `SYM` is not present in the selected dataset splits.

Official source: https://github.com/UniversalDependencies/UD_Persian-PerDT

## Model

`distilbert/distilbert-base-multilingual-cased`

The task is formulated as word-level sequence labeling. The first subword token receives the corresponding word-level POS label, while continuation subwords and special tokens are excluded from loss computation and word-level evaluation.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Maximum sequence length | 256 |
| Batch size | 8 |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Random seed | 42 |
| Frozen Transformer blocks | 3 (lower blocks) |

Both conditions use the same dataset splits, model architecture, and main training configuration.

## Experimental Results

| Metric | Full Fine-Tuning | Partial Freezing |
|---|---:|---:|
| Best Dev Accuracy | 96.17% | 95.48% |
| Test Accuracy | 96.36% | 95.78% |
| Test Macro F1 | 88.73% | 88.07% |
| Trainable Parameters | 134.75M | 113.48M |
| Training Loop + Validation + Checkpoint Time (seconds) | 779.50 | 674.81 |

Partial freezing reduced the measured training-loop, validation, and checkpoint-saving time by approximately **13.43%**.

This reduction came with a decrease of approximately **0.584 percentage points** in test accuracy and **0.658 percentage points** in macro F1.

The partial-freezing model retained approximately **84.22%** of the baseline model's trainable parameters. Freezing three of the six Transformer blocks does not imply that half of all model parameters are frozen, because embeddings and other model components remain trainable.

### Timing Definition

Runtime is measured using `time.perf_counter()` inside the `train_condition` function.

The reported time includes:

- The three-epoch training loop.
- Validation evaluation after each epoch.
- Saving the best model checkpoint during training.

It excludes:

- Dataset downloading and tokenization.
- Model, DataLoader, and optimizer initialization.
- Training-history CSV export.
- Checkpoint reloading.
- Final test evaluation.

Therefore, the reported values represent the measured training-loop, validation, and checkpoint-saving time, not the total end-to-end experiment runtime.

Runtime measurements depend on the hardware and software environment.

## Macro-F1 Sensitivity Analysis

The primary evaluation reports macro F1 across all 16 observed UPOS classes.

The `X` class has only one test instance. Its limited representation makes class-level performance estimates highly sensitive to individual predictions.

A supplementary sensitivity analysis was therefore conducted by excluding `X` from the unweighted mean of class-level F1 scores.

| Metric | Full Fine-Tuning | Partial Freezing |
|---|---:|---:|
| Macro F1 - All 16 Classes | 0.887309 | 0.880732 |
| Macro F1 - Excluding X | 0.946463 | 0.939448 |

Excluding `X` is supplementary only. The primary reported result remains macro F1 across all 16 observed classes.

See `results/macro_f1_sensitivity.csv` for the recorded values.

## Repository Structure

- `POS_Tagging_Partial_Freezing.ipynb`: Main executed experiment notebook
- `results/`: Evaluation metrics, training histories, model comparisons, class-level reports, and experiment metadata
- `reports/`: Detailed experimental record and analysis
- `requirements.txt`: General project dependencies
- `requirements-lock.txt`: Tested Windows environment snapshot
- `.gitignore`: Excludes local datasets, model checkpoints, temporary files, and environment-specific artifacts

Model checkpoints are stored locally and excluded from version control because of their file sizes. The tracked result files contain the evaluation metrics and other recorded experiment outputs.

## Environment and Reproducibility

The experiment was conducted using:

- Python 3.11.9
- PyTorch 2.14.1 + CUDA 13.0
- Transformers 5.23.0
- NumPy 2.4.6
- Pandas 3.1.1
- Scikit-learn 1.9.1
- NVIDIA GeForce RTX 3050 Laptop GPU (6 GB)

The notebook records experiment metadata, dataset checksums, training configuration, and evaluation outputs.

`requirements.txt` provides general dependency specifications. `requirements-lock.txt` records the tested Windows environment, including the CUDA-specific PyTorch package, and is not intended as a universal dependency specification for all platforms.

Downloaded raw datasets and trained model checkpoints are excluded from version control. They can be regenerated by running the notebook, subject to the availability of the original dataset and compatible software and hardware environments.

## Limitations

- Only one random seed was evaluated.
- Only one partial-freezing configuration was evaluated.
- The training set uses a 12,000-sentence subset rather than the complete training split.
- Class `X` has only two training examples, no validation examples, and one test example.
- Thirty-five validation words in one sentence were truncated at the maximum sequence length. No training or test words were truncated.
- The experiments cover only one language and treebank.
- Runtime measurements may vary across hardware and software environments.

The findings should therefore be interpreted as preliminary experimental evidence rather than a general conclusion about partial freezing.

## Future Work

Potential extensions include:

- Evaluating multiple freezing depths.
- Repeating experiments with multiple random seeds.
- Investigating class imbalance and rare POS categories.
- Measuring GPU memory consumption and peak resource usage.
- Evaluating different sequence lengths and batch sizes.
- Testing additional languages and treebanks.

## Detailed Experimental Record

See `reports/part1_experiment_record.md` for the detailed experimental setup, results, preliminary observations, limitations, next experiments, and reproducibility notes.

## Research Integrity

The reported metrics are based on the experiments conducted in this project. The executed notebook and accompanying result files provide the implementation and recorded evaluation outputs for inspection.

The findings are limited to the evaluated dataset subset, model configuration, freezing strategy, and random seed. No unexecuted experiment or estimated metric is presented as an observed result.