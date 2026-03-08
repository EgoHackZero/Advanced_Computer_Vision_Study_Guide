## 9. Network Architecture, Activation Functions, Learning Rate & Initialization, First vs. Second Order Methods

### Impact of Network Architecture

Deeper networks can represent more complex functions with fewer neurons by hierarchical feature composition. Wider networks have more parallel processing but require more parameters for the same abstraction. Skip connections (ResNet) ease training by allowing gradients to flow in very deep nets.

### Activation Functions

**Sigmoid** ($\sigma(x) = 1/(1 + e^{-x})$): Maps to $(0, 1)$. Saturates for large inputs → vanishing gradients. Not zero-centered.

**Tanh** ($\tanh(x) = 2\sigma(2x) - 1$): Maps to $(-1, 1)$. Zero-centered but still saturates at $\pm 1$ → vanishing gradients.

**ReLU** ($f(x) = \max(0, x)$): Piecewise linear. Does not saturate for positive values → mitigates vanishing gradient problem. Cheap to compute, promotes sparsity. Downside: neurons can "die" (always negative input → zero gradient). Variants like **Leaky ReLU** address this.

Modern nets use ReLU or variants in hidden layers. Sigmoid is reserved for output layers where a probabilistic (0–1) output is needed.

### Learning Rate

Too large → divergence or oscillation. Too small → slow convergence. Common practice: use learning rate schedules (decay over time) or adaptive rates (Adam). Try a range on log scale (e.g., $10^{-4}$ to $10^{-1}$) and find the largest that doesn't diverge.

### Initialization

**Symmetry breaking** — Random initialization ensures neurons develop differently.

**Variance propagation** — Too small weights → signals/gradients diminish. Too large → saturation or explosion. Schemes like **Xavier/Glorot** (for sigmoid/tanh) and **He initialization** (for ReLU) set variance to preserve activation magnitude across layers.

### First- vs. Second-Order Optimization

**First-order** (SGD, Momentum, Adam, RMSProp): Use gradient only. Cheap per step, scalable. The standard for deep learning.

**Second-order** (Newton's method, L-BFGS): Use Hessian (curvature) information for more informed steps. Converge in fewer iterations but computing/inverting the Hessian is intractable for large networks ($O(n^3)$ for $n$ parameters). Rarely used in modern deep learning.