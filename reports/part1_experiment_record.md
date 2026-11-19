# Part 1: Experimental Record
## Partial Freezing of DistilBERT for Persian POS Tagging

**Researcher:** Kiarash Ghaderi
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)
**Python:** 3.11.9
**PyTorch:** 2.14.1+cu130
**Transformers:** 5.23.0
**Random Seed:** 42

---

## 1. Experimental Setup

The experiments were conducted on the Universal Dependencies (UD) Persian PerDT treebank using a multilingual DistilBERT model for word-level Part-of-Speech (POS) tagging.

The experimental configuration was kept consistent across both model conditions to support a controlled comparison and minimize potential confounding factors.

| Configuration | Value |
|---|---|
| Dataset | UD Persian PerDT |
| Training Subset | 12,000 sentences |
| Validation Set | 1,456 sentences |
| Test Set | 1,455 sentences |
| Pretrained Model | `distilbert/distilbert-base-multilingual-cased` |
| Maximum Sequence Length | 256 |
| Batch Size | 8 |
| Training Epochs | 3 |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Random Seed | 42 |

The same dataset splits, preprocessing procedure, optimization settings, and evaluation protocol were used for both configurations.

## 2. Model Configurations

Two fine-tuning strategies were evaluated.

**Baseline: Full Fine-Tuning**

All model parameters, including the embeddings, all six Transformer blocks, and the classification head, were trainable during fine-tuning.

**Partial Freezing**

The lower three Transformer blocks were frozen, while the embeddings, upper three Transformer blocks, and classification head remained trainable.

This configuration was designed to investigate whether restricting parameter updates in the lower Transformer layers could reduce training time while maintaining competitive POS-tagging performance.

The lower layers were selected for freezing to preserve a portion of the pretrained representations while allowing the upper layers to adapt to the target POS-tagging task.

## 3. Experimental Results

The following table summarizes the validation and test performance, trainable parameter counts, and measured training time for both configurations.

| Metric | Full Fine-Tuning | Partial Freezing |
|---|---:|---:|
| Best Development Accuracy | 96.17% | 95.48% |
| Test Accuracy | 96.36% | 95.78% |
| Test Macro-F1 | 88.73% | 88.07% |
| Trainable Parameters | 134,746,384 | 113,482,768 |
| Training Loop + Validation + Checkpoint Time (seconds) | 779.50 | 674.81 |

### 3.1. Training-Time Measurement

Training time was measured using Python's `time.perf_counter()` within the `train_condition` function.

The measurement includes:

- The three-epoch training loop.
- Validation evaluation after each epoch.
- Saving the best-performing checkpoint during training.

The measurement excludes:

- Dataset downloading and tokenization.
- Model, DataLoader, optimizer, and scheduler initialization.
- Training-history CSV export.
- Checkpoint reloading.
- Final test-set evaluation.

Therefore, the reported values represent the measured training-loop, validation, and checkpoint-saving time rather than the total end-to-end experiment runtime.

## 4. Comparative Analysis and Preliminary Observations

Partial freezing reduced the measured training-loop and validation-evaluation time, including checkpoint saving, from 779.50 seconds to 674.81 seconds, corresponding to an approximate **13.43% reduction**.

However, this improvement in measured training efficiency was accompanied by a modest reduction in predictive performance:

- Test accuracy decreased by approximately **0.584 percentage points**.
- Test Macro-F1 decreased by approximately **0.658 percentage points**.

The partial-freezing configuration retained approximately **84.22% of the model's total parameters as trainable**, corresponding to approximately 15.78% of the total parameters being frozen.

Although three of the six Transformer blocks were frozen, this did not correspond to freezing half of the entire model's parameters. The embeddings, upper Transformer blocks, and classification head remained trainable.

These results suggest that freezing the lower three Transformer blocks reduced the measured training time, but the resulting efficiency improvement did not fully preserve the predictive performance of full fine-tuning.

One possible explanation is that allowing all Transformer layers to adapt provides additional flexibility for learning task-specific representations. However, this interpretation remains a hypothesis and requires further experiments to establish the relationship between freezing depth and POS-tagging performance.

Overall, the results indicate a trade-off between measured training efficiency and predictive performance under the evaluated configuration.

## 5. Dataset Characteristics and Evaluation Considerations

The UPOS class `X` is extremely underrepresented in the selected dataset splits:

- Training split: 2 instances.
- Validation split: 0 instances.
- Test split: 1 instance.

Both training instances were included in the selected 12,000-sentence training subset. Nevertheless, this extremely limited representation does not provide sufficient evidence for reliable learning or meaningful class-level conclusions for this category.

Additionally, 35 validation words within one sentence were truncated because of the maximum sequence length of 256 tokens. No words in the test set were truncated.

These characteristics should be considered when interpreting the reported class-level and aggregate evaluation metrics, particularly the unweighted Macro-F1 score.

## 6. Experimental Limitations

The current experiments have several limitations:

- Only one random seed was evaluated.
- Only one partial-freezing configuration, with the lower three Transformer blocks frozen, was investigated.
- The training experiment used a 12,000-sentence subset rather than the complete training split.
- The extremely limited representation of certain POS categories may affect macro-averaged evaluation metrics.
- Only one language and treebank were considered.
- GPU memory consumption was not systematically measured or compared.
- The reported timing covers the training loop, validation, and checkpoint saving, rather than the complete end-to-end experimental runtime.

Consequently, the findings should be interpreted as preliminary experimental observations rather than general conclusions about partial freezing.

## 7. Future Experiments

The following experiments are proposed to extend the current investigation:

1. Evaluate different freezing depths, including freezing fewer or more Transformer blocks.
2. Repeat the main experiments using multiple random seeds to assess result stability.
3. Investigate the impact of class imbalance and rare POS categories.
4. Analyze class-level performance differences to identify categories most affected by partial freezing.
5. Measure and compare GPU memory consumption and computational efficiency.
6. Investigate whether the observed efficiency-performance trade-off remains consistent under larger training subsets.
7. Examine whether layer-wise freezing strategies can preserve task-relevant representations while reducing computational overhead.

These experiments would help determine whether the observed trade-off is specific to the selected freezing configuration or reflects a more general pattern in multilingual Transformer fine-tuning.

## 8. Reproducibility and Project Artifacts

The experiment configuration, dataset information, hardware specifications, and relevant metadata are recorded in `results/run_metadata.json`.

The repository also retains the following experimental artifacts:

- Executed Jupyter Notebook containing the experimental workflow.
- Training histories for both model configurations.
- Model comparison results.
- Class-level classification reports.
- UPOS split statistics.
- Macro-F1 sensitivity analysis.
- Experimental metrics and run metadata.

The trained model checkpoints are retained locally and excluded from the Git repository because of their large file sizes.

The dependency specifications are provided through `requirements.txt` and `requirements-lock.txt` to support environment setup and reproducibility.

## 9. Research Integrity

All reported metrics and observations are based on the actual experiments conducted in the local environment.

No unexecuted experiment, estimated metric, or hypothetical result is presented as an observed experimental finding.

AI assistance was used for implementation support, debugging, and organization of the experimental documentation. The researcher remains responsible for verifying the implementation, validating the reported results, and critically reviewing the scientific interpretation before submission.

## 10. Macro-F1 Sensitivity Analysis

The test set contains only one instance of the `X` UPOS class. Since a single observation provides an extremely limited basis for estimating class-level performance, an additional sensitivity analysis was conducted by excluding `X` from the unweighted mean of class-level F1 scores.

| Metric | Full Fine-Tuning | Partial Freezing |
|---|---:|---:|
| Macro-F1 (All 16 Observed Classes) | 0.887309 | 0.880732 |
| Macro-F1 (Excluding `X`) | 0.946463 | 0.939448 |

The primary reported Macro-F1 results include all 16 observed UPOS classes. The values excluding `X` are presented exclusively as a supplementary sensitivity analysis and do not replace the primary evaluation metric.

This additional analysis illustrates the influence of an extremely rare class on the unweighted average of class-level F1 scores. The supplementary results should not be interpreted as an alternative primary evaluation metric.

The corresponding results are available in `results/macro_f1_sensitivity.csv`.

## 11. Research and Implementation Time

The following table provides a retrospective estimate of the active research and implementation time spent on Part 1. These estimates represent working time allocated to the different project activities rather than the model's execution time.

| Activity | Estimated Time |
|---|---:|
| Literature review and problem understanding | 3 hours |
| Dataset inspection and preprocessing | 3 hours |
| Model implementation and configuration | 5 hours |
| Model training and experimentation | 4 hours |
| Evaluation and result analysis | 3 hours |
| Documentation and reproducibility preparation | 4 hours |
| **Total** | **22 hours** |

**Time-estimation note:** The values above are retrospective estimates of active research and implementation time. They are not automatically measured working hours and should be adjusted if they do not accurately reflect the researcher's actual effort. Unattended model execution time is not counted as active working time.

The research and implementation time reported in this section is separate from the model training time measured in Section 3.
