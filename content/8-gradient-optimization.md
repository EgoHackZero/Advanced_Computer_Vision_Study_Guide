## 8. Gradient-Based Optimization: SGD, Backpropagation, Loss Functions, Batches, and Momentum

### Stochastic Gradient Descent (SGD)

SGD is an iterative method for optimizing an objective function using a subset of data (often a mini-batch) to estimate the gradient each step. This introduces randomness but greatly speeds up training for large datasets. Mini-batch SGD (batch sizes like 32, 64) is the standard in deep learning, smoothing noise and benefiting from parallelism.

### Backpropagation

Backpropagation efficiently computes gradients of the network's loss with respect to every weight via the chain rule. First a **forward pass** computes output and loss, then a **backward pass** propagates gradients layer by layer, computing $\partial L / \partial W$ for each weight. This allows information from the cost to flow backwards through the network. Backprop computes all gradients in time linear in network size — essential for training deep networks.

### Error / Loss / Cost Functions

- **Loss function** — Measures discrepancy between prediction and target for a single example (e.g., MSE, Cross-Entropy).
- **Cost function** — The average of losses over the entire training set or batch: $J(w) = \frac{1}{N}\sum_i L^{(i)}(w)$. This is the objective SGD minimizes.
- **Error** — Sometimes used interchangeably with loss, sometimes refers to raw metrics like misclassification count.

### Batch Learning

- **Full-batch gradient descent** — Uses entire training set per update. Precise gradient but slow.
- **Mini-batch SGD** — Process e.g., 32 samples, compute average gradient, update weights. Standard compromise.
- Larger batch → more accurate gradient direction (less variance). Smaller batch → beneficial noise that can escape sharp minima and often generalizes better.

### Momentum

Momentum adds inertia to weight updates, maintaining a velocity vector:

$$v \leftarrow \beta v + \eta \nabla_w L$$
$$w \leftarrow w - v$$

Where $\beta \in [0, 1)$ is the momentum coefficient (commonly 0.9). Effects:
- If gradients consistently point in the same direction → updates **accelerate**
- If gradients oscillate (e.g., narrow valley) → momentum **damps** the back-and-forth
- Result: faster convergence, ability to overcome small local minima, reduced oscillation

### Nesterov Momentum (NAG)

A variant where the gradient is evaluated at the "lookahead" position instead of the current position:

$$v \leftarrow \beta v + \eta \nabla_w L(w - \beta v)$$
$$w \leftarrow w - v$$

The key difference: standard momentum computes the gradient at $w$, then applies momentum. Nesterov first "looks ahead" to where momentum would take us ($w - \beta v$), then computes the gradient there. This often gives better convergence because the gradient correction is more informed — it anticipates where the parameters are heading.