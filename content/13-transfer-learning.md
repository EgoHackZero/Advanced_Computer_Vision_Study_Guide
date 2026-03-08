## 13. Transfer Learning and Dataset Splits

### Transfer Learning

A model trained on one task is reused or adapted for a new, related task. The pre-trained model provides good feature extractors (edges, textures, shapes) that are generic across many vision tasks.

### Settings Based on Dataset Size and Similarity

| | **Small target dataset** | **Large target dataset** |
|---|---|---|
| **Similar to source** | Use network as fixed feature extractor — train only a linear classifier on top-layer features. Do NOT fine-tune (overfitting risk). | Fine-tune entire model with small learning rate. |
| **Different from source** | Extract features from **earlier/middle layers** (top layers are too domain-specific). Train linear classifier on those. | Fine-tune entire model more aggressively, possibly from scratch using pre-trained weights as initialization. |

> [!important] CS231n convention
> The key insight is: with small data, **never fine-tune** — only train a classifier. With different domains, use **earlier** layer features (not top layers). These are the standard expected answers on exams following the CS231n curriculum.

### Why Transfer Learning Works

Early/mid layers learn universal features (edge detectors, textures). Pre-trained weights provide a good initialization — the model starts in a favorable region of parameter space, requiring fewer examples and less training time.

### Train / Validation / Test Splits

**Training set** — Used to fit model parameters. Model "sees" and learns from these examples.

**Validation set** — Not used in training. Used for hyperparameter tuning, model selection, and early stopping. Acts as proxy for test data during development.

**Test set** — Held out until final evaluation. Provides unbiased assessment of the model's generalization. Only used once after all tuning is complete.

Common splits: 80/10/10 or 70/15/15. Key rule: the test set must remain untouched during model development. Using test data to make decisions invalidates the generalization estimate.