## 10. Regularization (L1 vs. L2, Dropout), Dropout at Inference, and Ensemble Learning

### L1 vs. L2 Regularization

**L2 (Ridge / Weight Decay):** Penalty $\frac{\lambda}{2}\sum_i w_i^2$. Shrinks weights uniformly. Results in many small but non-zero weights. Smoother models, stable optimization.

**L1 (Lasso):** Penalty $\lambda \sum_i |w_i|$. Encourages sparsity — drives some weights to exactly zero. Performs feature selection. Gradient is constant for non-zero weights with sharp corners at zero.

L1 tends to shrink some weights all the way to zero (sparse solution), whereas L2 shrinks weights more uniformly without making most of them exactly zero.

### Dropout

During training, dropout randomly sets a fraction of neurons to zero on each forward pass (typically $p = 0.5$ keep probability). Forces distributed representations — the network cannot rely on any single neuron.

**Inverted Dropout:** Instead of scaling at test time, scale up surviving neurons by $1/p$ during training. Expected output: $p \cdot (a/p) + (1-p) \cdot 0 = a$. At test time, use the network as-is — no special scaling needed. Most frameworks implement inverted dropout by default.

**At inference:** No dropout is applied. All neurons are active. If inverted dropout was used, no scaling is needed.

> [!warning] PyTorch convention
> In PyTorch, the `p` parameter in `nn.Dropout(p)` is the **drop** probability (not keep probability). So `nn.Dropout(0.5)` drops 50% of neurons and scales surviving ones by $1/(1-0.5) = 2$. This is numerically identical to CS231n's convention where $p$ is the keep probability and scaling is $1/p$ — just opposite naming.

### Ensemble Learning — Boosting vs. Bagging

**Bagging (Bootstrap Aggregating):**
- Train models independently in parallel on bootstrap samples
- Average predictions (regression) or vote (classification)
- Main effect: **reduces variance**
- Example: Random Forest (bagging of decision trees)

**Boosting:**
- Train models sequentially, each correcting mistakes of the previous
- Reweight misclassified examples, increase focus on "hard" cases
- Combine with weighted vote ($\alpha_m = \log((1 - err_m)/err_m)$)
- Main effect: **reduces bias** (and also variance)
- Examples: AdaBoost, Gradient Boosting (XGBoost)

Key difference: Bagging = parallel, independent, reduces variance. Boosting = sequential, dependent, reduces bias.