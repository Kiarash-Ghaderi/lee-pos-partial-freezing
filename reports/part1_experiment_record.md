# Part 1: Experimental Record
## Partial Freezing of DistilmBERT for Persian POS Tagging

**Researcher:** Kiarash Ghaderi  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB)  
**Python:** 3.11.9  
**PyTorch:** 2.14.1+cu130  
**Transformers:** 5.23.0  
**Seed:** 42

## 1. Experimental Setup

- Dataset: UD Persian PerDT
- Training subset: 12,000 sentences
- Validation: 1,456 sentences
- Test: 1,455 sentences
- Model: distilbert/distilbert-base-multilingual-cased
- Maximum sequence length: 256
- Batch size: 8
- Epochs: 3
- Optimizer: AdamW
- Learning rate: 2e-5
- Weight decay: 0.01

## 2. Model Conditions

**Baseline:** All encoder layers, embeddings, and the classification head were trainable.

**Partial Freezing:** The lower three Transformer blocks were frozen. Embeddings, upper three Transformer blocks, and the classification head remained trainable.

## 3. Experimental Results

| Metric | Baseline | Partial Freezing |
|---|---:|---:|
| Best Dev Accuracy | 96.17% | 95.48% |
| Test Accuracy | 96.36% | 95.78% |
| Test Macro F1 | 88.73% | 88.07% |
| Trainable Parameters | 134,746,384 | 113,482,768 |
| Training and Validation Time (seconds) | 779.50 sec | 674.81 sec |

The reported training time is measured using time.perf_counter() inside train_condition. It includes the three-epoch training loop, validation evaluation after each epoch, and best-checkpoint saving. It excludes dataset downloading, tokenization, model and optimizer initialization, training-history export, checkpoint reloading, and final test evaluation.

## 4. Preliminary Observations

Partial freezing reduced the measured training-loop and validation-evaluation time by approximately 13.43%.

Test accuracy decreased by approximately 0.584 percentage points, while macro-F1 decreased by approximately 0.658 percentage points.

The partial-freezing condition updated approximately 84.22% of the model parameters. Freezing three of six Transformer blocks did not mean that half of the complete model parameters were frozen, because the embeddings and other components remained trainable.

## 5. Dataset Limitations

The UPOS label X occurs only twice in the complete training split, zero times in validation, and once in test.

The selected training subset includes both training examples, but the number of examples is insufficient for reliable learning or meaningful class-level conclusions.

Furthermore, 35 validation words across one sentence were truncated at the maximum sequence length. No test words were truncated.

## 6. Experimental Limitations

- Only one random seed has been evaluated.
- Only one freezing configuration has been evaluated.
- The training subset is smaller than the complete training split.
- The results should be considered preliminary.
- Further experiments are necessary before making general claims about partial freezing.

## 7. Next Experiments

1. Compare different freezing depths.
2. Repeat the main comparison with multiple random seeds.
3. Investigate the effect of class imbalance.
4. Analyze which POS categories are most affected by freezing.
5. Compare training efficiency and memory consumption.

## 8. Reproducibility

Dataset checksums and hardware information are recorded in `results/run_metadata.json`.

Training histories, class-level reports, model comparison tables, and the executed notebook are retained as project artifacts.

The trained model checkpoints are stored locally and excluded from the Git repository because of their file sizes.

## 9. Research Integrity

These results are from the actual local experiment. No unexecuted experiment or estimated metric is presented as an observed result.

AI assistance was used for implementation support and organization of the experimental record. The researcher should independently verify and critically revise the scientific interpretation before submission.

## Macro-F1 Sensitivity Analysis

The test set contains only one instance of the `X` UPOS class. Because
a single example provides a very limited basis for estimating class-level
performance, a supplemental sensitivity analysis was calculated by
excluding `X` from the unweighted mean of class-level F1 scores.

| Metric | Baseline | Partial Freezing |
|---|---:|---:|
| Macro F1 - All 16 classes | 0.887309 | 0.880732 |
| Macro F1 - Excluding X | 0.946463 | 0.939448 |

This analysis does not replace the primary 16-class macro-F1 result.
It is reported separately to show how the rare `X` class affects the
unweighted class-average metric. The values are saved in
`results/macro_f1_sensitivity.csv`.
