## 11. Loss Functions, Data Augmentation, and Data Preprocessing

### Loss Functions

**MSE (Mean Squared Error):** Average of squared differences. Heavily penalizes large errors → sensitive to outliers. Used for regression.

**MAE (Mean Absolute Error):** Uses absolute differences. Equal weight to all errors regardless of magnitude → less sensitive to outliers. Useful when robustness is needed.

**Hinge Loss:** $L = \max(0, 1 - y \cdot f)$. Zero when correctly classified with margin ≥ 1. Encourages maximum-margin separation. Used with SVMs.

**Binary Cross-Entropy:** $L = -[y \log p + (1-y)\log(1-p)]$. Heavily penalizes confident wrong predictions. Preferred for probabilistic classification (logistic regression, neural nets with sigmoid output).