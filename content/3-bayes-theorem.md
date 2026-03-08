## 3. Bayes' Theorem in Classification

### Bayes' Theorem

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

It updates the probability of class $C$ given observed evidence $X$ by weighing the likelihood of $X$ under $C$ with the prior belief in $C$.

### Example: Cat vs. Dog Classification

Suppose we classify an image as Cat or Dog based on feature $X$ = "pointy ears detected."

- Prior: $P(\text{Cat}) = 0.1$, $P(\text{Dog}) = 0.9$
- Likelihood: $P(X|\text{Cat}) = 0.8$, $P(X|\text{Dog}) = 0.2$

Applying Bayes' rule:

$$P(\text{Cat}|X) \propto 0.8 \times 0.1 = 0.08$$
$$P(\text{Dog}|X) \propto 0.2 \times 0.9 = 0.18$$

After normalization: $P(\text{Cat}|X) = 0.08/0.26 \approx 31\%$ and $P(\text{Dog}|X) \approx 69\%$.

**Interpretation:** Even though pointy ears are more likely on cats than dogs, the dog prior was so much higher that the image is still more likely a dog. Bayes' rule balances feature evidence with prior expectations.

### Use in Object Classification

In a Naïve Bayes classifier for BoW features, assuming independence of visual words given the class:

$$\hat{C} = \arg\max_C \; P(C) \prod_{n=1}^{N} P(w_n | C)$$

Where $P(w_n|C)$ are learned likelihoods (e.g., a face class gives high probability to codewords for eyes or noses) and $P(C)$ reflects class frequency. Bayes' theorem is fundamental for probabilistic classification — it provides a principled way to go from $P(X|C)$ to predictions $P(C|X)$.

> [!warning] Practical essentials for exams
> **Laplace smoothing:** Without it, any visual word with zero probability for a class zeroes the entire product. Add a small constant (e.g., +1) to all word counts.
> **Log-space computation:** Multiplying many small probabilities causes numerical underflow. In practice, take the log: $\log P(C|X) \propto \log P(C) + \sum_n \log P(w_n|C)$ — this turns products into sums.
> **Multinomial vs. Bernoulli:** Multinomial NB uses word counts; Bernoulli NB uses binary presence/absence of words.