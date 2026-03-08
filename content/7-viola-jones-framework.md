## 7. Viola-Jones Framework: Integral Images, Haar Features, AdaBoost, and Cascade

### Haar-Like Features

Viola-Jones uses simple rectangular features reminiscent of Haar wavelets. A Haar feature value is the difference of pixel intensity sums in adjacent rectangular regions. These detect edges, lines, and center-surround patterns. For example, the bridge of the nose appears as a bright region flanked by darker eyes. Over **180,000** possible features can be placed in a 24×24 detection window (the original Viola-Jones CVPR 2001 paper states "over 180,000").

### Integral Image

The integral image $II(x, y)$ stores the sum of all pixels above and to the left of $(x, y)$. Once computed, the sum of any axis-aligned rectangular region can be obtained in **constant time**:

$$\text{Sum}(ABCD) = II(D) + II(A) - II(B) - II(C)$$

Each Haar feature is evaluated with just a few array lookups, independent of rectangle size. This enables scanning thousands of subwindows in real-time.

### AdaBoost for Feature Selection and Classification

AdaBoost (Adaptive Boosting) takes many weak classifiers (each a single Haar feature threshold test) and combines them into a weighted "strong" classifier. The process:

1. Iteratively select the feature that best classifies training data (faces/non-faces) with minimal error.
2. Reweight training examples, focusing on misclassified ones.
3. Repeat for many rounds.
4. Final classifier: $H(x) = \sum_i \alpha_i h_i(x)$ where $h_i$ are selected Haar features.

AdaBoost selects only a few hundred features out of 180,000+ candidates, drastically reducing computation while identifying the most discriminative features.

### Attentional Cascade

The cascade arranges boosted classifiers into sequential stages:

- **Stage 1** — Very few features (e.g., 2–5). High recall, many false positives. Any window failing is immediately rejected.
- **Stage 2** — More features (e.g., 10). Further scrutiny of passing windows.
- **Subsequent stages** — Increasingly more features, applied to progressively fewer candidate windows.

The majority of windows are rejected by early, cheap stages. Only face-like regions undergo full evaluation. This enabled Viola-Jones to scan a 384×288 image in ~0.067 seconds on 2001 hardware (700 MHz Pentium III), achieving real-time face detection.