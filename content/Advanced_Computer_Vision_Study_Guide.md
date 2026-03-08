## 1. Hierarchical Model of Vision and Object Categorization

### Hierarchical Visual Model

The primate visual system is organized hierarchically, with neurons in early stages (e.g., V1) detecting simple features (edges), and later stages (V4, IT) responding to complex shapes and objects. Hubel and Wiesel's classic experiments showed that vision is constructed through a cascade of cells in a layered hierarchy. Each layer combines inputs from earlier layers, building more abstract, invariant features.

This inspired hierarchical computer vision models like Fukushima's **Neocognitron** (the precursor to CNNs), which used cascaded simple and complex cells to recognize patterns. Modern deep CNNs exploit a similar layered hierarchy: lower layers learn edges and textures, higher layers learn object parts and categories, mirroring the ventral stream of human vision.

### Methods Utilizing Hierarchy

Early vision models (e.g., HMAX by Riesenhuber & Poggio) explicitly stacked feature extraction layers to mimic the visual cortex. Today's deep networks inherently implement hierarchical feature learning — each convolutional layer extracts higher-level features from the previous layer's outputs. Hierarchical models achieve specificity to object details while gaining invariance to low-level variations.

### Object Categorization — 7 Key Challenges

1. **Viewpoint Variation** — Objects appear different from new angles (e.g., a mug's handle visible from the side but not from above). A vision system must handle perspective changes.
2. **Scale Variation** — Objects have dramatically different pixel sizes depending on distance. A detection method must identify objects at various scales.
3. **Illumination Changes** — Lighting alters pixel intensities and shadows. An object can appear bright or dark yet must be recognized as the same entity.
4. **Deformation** — Many objects (animals, humans) are non-rigid or articulated. The model must tolerate shape deformations.
5. **Occlusion** — Objects are often partially blocked by others. Even with only some parts visible, the system should detect the object.
6. **Background Clutter** — The scene may contain complex backgrounds. The target object's features must be distinguished from similar textures in the background.
7. **Intra-class Variation** — Instances of the same category can look very different (e.g., armchair vs. stool). The model must learn essential category features despite color, size, and style differences.

These challenges make object categorization extremely hard. In practice, robust recognition systems rely on large training data and invariant features. The hierarchical architecture of modern CNNs helps by learning progressively more invariant features (edge detectors → part detectors → whole-object detectors), but significant variation and occlusion remain active research problems.

---

## 2. Generative vs. Discriminative Models, Appearance-Only vs. Position Models, and Bag-of-Words

### Generative vs. Discriminative Representations

A **generative model** learns the joint distribution of features and class labels, modeling how image features are generated for each object class. It can compute likelihoods $P(\text{features}|\text{class})$. Examples: Naïve Bayes, probabilistic graphical models.

A **discriminative model** directly learns the decision boundary or conditional distribution $P(\text{class}|\text{features})$. It focuses only on classification accuracy, often yielding better performance since it directly optimizes the predictive function. Examples: SVMs, neural networks.

In summary: discriminative models classify data points without providing a model of how the data were generated, whereas generative models provide a model of how data are generated in terms of a probabilistic model.

### Appearance-Only vs. Appearance + Position Models

An **appearance-only model** represents an object using only the visual appearance of local patches or features, ignoring their spatial arrangement. A prime example is the Bag-of-Words image model. It captures texture and part appearance frequencies but not where those parts occur. Appearance-only models are invariant to geometry but risk confusion between objects with similar textures arranged differently.

**Appearance + Position models** explicitly account for the spatial configuration of features. For instance, a deformable part model might learn both what each part looks like and their relative positions (e.g., for a face: eyes above nose, nose above mouth). Including spatial layout greatly improves discrimination: exploiting spatial co-occurrences of visual words improves classification accuracy.

### Bag-of-Words (BoW) Model

The BoW model is a classic appearance-only representation for categorization, analogizing images to documents:

1. **Feature Detection** — Detect salient local regions (e.g., keypoints).
2. **Feature Description** — Describe each patch by a descriptor (e.g., SIFT 128-dim vector).
3. **Codebook Generation** — Cluster descriptors from training images into $K$ visual "words" (using k-means). The cluster centers form a codebook.
4. **Quantization** — Map each patch's descriptor to the nearest codebook word.
5. **Histogram Encoding** — Represent each image by a histogram (length $K$ vector) counting how many patches fell into each visual word bin.

The order and position of patches are discarded — the image is just a "bag" of visual words. For classification, one can use generative approaches (e.g., Naïve Bayes on BoW counts) or discriminative classifiers (SVM, etc.). The BoW model is simple and robust to position, but ignores spatial relationships. Extensions like **spatial pyramids** (Lazebnik et al., 2006) partly address this by computing BoW histograms over subregions of the image.

> [!tip] Important additions for exams
> **TF-IDF weighting** can be applied to visual words — down-weighting common visual words (like uniform patches) that appear in many images and up-weighting distinctive ones. **Fisher Vectors** and **VLAD** (Vectors of Locally Aggregated Descriptors) are more powerful successors to BoW that encode not just word counts but also first/second-order statistics of descriptor residuals relative to cluster centers.

---

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

---

## 4. SIFT Algorithm: Scale Space, Keypoint Detection, Filtering, Localization

### Overview

The Scale-Invariant Feature Transform (SIFT) detects and describes local image features robust to scale and rotation changes.

### 1. Scale-Space Construction

The input image is analyzed across multiple scales. SIFT creates an image pyramid by progressively smoothing with Gaussians and subsampling. SIFT approximates the Laplacian-of-Gaussian (LoG) with a **Difference-of-Gaussians (DoG)** filter for efficiency. Each octave is blurred with Gaussians of increasing $\sigma$, and adjacent blurred images are subtracted to produce DoG images. LoG acts as a blob detector; DoG is an efficient approximation.

### 2. Keypoint Detection (Extrema in Scale-Space)

SIFT locates keypoints by finding local extrema (maxima or minima) in the DoG volume. Each pixel in a DoG image is compared to its 8 neighbors at the same scale and the 9 corresponding neighbors in the scale above and below. If it is the largest or smallest among those 26 neighbors, it is a potential keypoint, giving initial $(x, y, \sigma)$ estimates.

### 3. Keypoint Localization via Taylor Expansion

For each candidate, a 3D quadratic function is fit (Taylor expansion of the DoG) to interpolate the keypoint's location in scale and space with sub-pixel accuracy.

### 4. Contrast Thresholding

If the interpolated DoG value at the keypoint is below a threshold (e.g., 0.03 in Lowe's paper), the point is discarded as too low-contrast (unstable against noise).

### 5. Edge Response Elimination

DoG extrema can occur along edges (poor localization). SIFT uses the Hessian matrix to compute principal curvatures. If the ratio of largest to smallest eigenvalue exceeds a threshold (10 in the paper), the candidate is deemed edge-like and is dropped. In practice, a ratio of the trace to determinant of the Hessian is used.

### 6. Orientation Assignment

Each keypoint is assigned a consistent orientation based on local image gradient directions. SIFT computes gradient magnitude and orientation in the keypoint's neighborhood and creates a 36-bin orientation histogram weighted by gradient magnitude and a Gaussian window. The dominant peak is chosen as the keypoint's orientation. Additional keypoints are created for other strong peaks (within ~80% of max). This provides **rotation invariance**.

### 7. Keypoint Descriptor

SIFT builds a 128-dimensional descriptor vector for the local patch around each keypoint (see Section 5 for full details).

After these steps, each keypoint has a location $(x, y)$, scale, orientation $\theta$, and descriptor — highly distinctive and invariant to scaling, rotation, and moderately invariant to viewpoint and illumination changes.

---

## 5. Keypoint Descriptors, Orientation & Scale Invariance, and Generalized Hough Transform

### Scale & Orientation Invariance

**Scale invariance** — each keypoint is detected at an inherent scale ($\sigma$ of the DoG extremum), so the same physical feature is found whether the object is near or far.

**Orientation invariance** — achieved by assigning a dominant orientation and rotating the patch before descriptor extraction. The SIFT feature for a corner looks the same regardless of image rotation, because the descriptor is computed relative to the keypoint's dominant orientation.

Each SIFT keypoint is a 4-tuple $(x, y, s, \theta)$ — location, scale $s$, and orientation $\theta$.

### Descriptor Construction

A $16 \times 16$ neighborhood (in the keypoint's scale) is rotated to the keypoint's orientation and divided into a $4 \times 4$ grid of subregions. In each subregion, an orientation histogram of 8 bins (360°) is computed from gradient magnitudes. This yields $4 \times 4 \times 8 = 128$ elements. The descriptor is normalized to unit length, then values are **thresholded at 0.2** (to reduce the influence of large gradient magnitudes caused by illumination changes), and then renormalized again.

> [!tip] Exam tip: Matching
> When matching SIFT descriptors, Lowe's **nearest-neighbor distance ratio test** is used: a match is accepted only if the ratio of the nearest to second-nearest neighbor distance is below **0.8**. This eliminates ambiguous matches effectively.

### Generalized Hough Transform (GHT) for Object Localization

The GHT uses SIFT matches to vote for object presence via a voting scheme:

1. **Database building** — For a target object, SIFT keypoints are extracted from reference images. Each model keypoint stores its relative location to the object's reference point.
2. **Matching** — Input image SIFT keypoints are compared against the database via nearest-neighbor in descriptor space.
3. **Voting** — Each match casts a vote in a Hough accumulator for the hypothesized object pose (position, scale, orientation). For example, if a model keypoint was 50px right and 20px up from the object's center, a match at $(x, y)$ votes for center at $(x - 50, y + 20)$, adjusted for scale/orientation.
4. **Clustering** — If enough keypoints agree on approximately the same pose, a consistent cluster is identified. A cluster with ≥ 3 matches is considered a strong hypothesis.
5. **Verification** — A least-squares solution verifies that the cluster can be explained by a single affine transform. Outliers are discarded and a probability is computed.

This approach was a breakthrough in specific object recognition — even in cluttered scenes, an object could be localized based on multiple keypoint correspondences and a voting scheme for pose consistency.

---

## 6. SURF vs. SIFT, Hessian Blob Detection, and SURF's Haar Wavelets

### SURF vs. SIFT — Key Differences

| Aspect | SIFT | SURF |
|---|---|---|
| **Keypoint Detection** | DoG extrema | Hessian matrix determinant (Fast Hessian) |
| **Descriptor Length** | 128-dimensional | 64-dimensional |
| **Orientation Assignment** | Gradient histograms | Haar wavelet responses in x, y |
| **Speed** | Slower | Faster (integral images, shorter descriptors) |
| **Robustness** | Slightly more reliable in extreme conditions | Trades some accuracy for speed |

SURF is generally faster in both detection and description due to integral images and shorter descriptors, making it suitable for real-time applications when SIFT was too slow.

### Blob Definition and Hessian Detection

A **blob** is an image region that is roughly uniform or differs in intensity/color from its surroundings. The Hessian matrix of image intensity is:

$$H(x, y) = \begin{pmatrix} L_{xx} & L_{xy} \\ L_{xy} & L_{yy} \end{pmatrix}$$

The determinant $\det(H) = L_{xx}L_{yy} - (L_{xy})^2$ is high when there is strong curvature in orthogonal directions — precisely the signature of a blob (both eigenvalues large and of same sign).

**Hessian properties:**
- Rotation invariant (eigenvalues are rotation invariant)
- Responds to blob intensity contrast
- Computable efficiently with box filter approximations using integral images — detection speed is independent of blob size

SURF's Fast Hessian applies box-filter approximations at multiple scales (9×9, 15×15, 21×21, etc.) and finds maxima of filter response.

### SURF's Haar Wavelet Descriptor

1. Take a square window (typically 20×20 pixels) centered on the keypoint, oriented to dominant orientation.
2. Divide into a $4 \times 4$ grid of subregions (each 5×5 pixels).
3. Within each subregion, compute Haar wavelet responses: $\sum dx$, $\sum dy$, $\sum |dx|$, $\sum |dy|$.
4. Each subregion yields a 4-dimensional vector. Concatenate: $16 \times 4 = 64$ dimensions.
5. Normalize the vector.

Haar wavelets are related to box filters approximating first-order Gaussian derivatives. SURF's descriptor is compact and fast to match.

---

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

---

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

---

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

---

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

---

## 11. Loss Functions, Data Augmentation, and Data Preprocessing

### Loss Functions

**MSE (Mean Squared Error):** Average of squared differences. Heavily penalizes large errors → sensitive to outliers. Used for regression.

**MAE (Mean Absolute Error):** Uses absolute differences. Equal weight to all errors regardless of magnitude → less sensitive to outliers. Useful when robustness is needed.

**Hinge Loss:** $L = \max(0, 1 - y \cdot f)$. Zero when correctly classified with margin ≥ 1. Encourages maximum-margin separation. Used with SVMs.

**Binary Cross-Entropy:** $L = -[y \log p + (1-y)\log(1-p)]$. Heavily penalizes confident wrong predictions. Preferred for probabilistic classification (logistic regression, neural nets with sigmoid output).

**Categorical Cross-Entropy (multi-class):** Generalization of BCE for $C$ classes with softmax output: $L = -\sum_{c=1}^{C} y_c \log(\hat{p}_c)$, where $y_c$ is 1 for the true class and 0 otherwise, and $\hat{p}_c = \text{softmax}(z_c) = e^{z_c} / \sum_j e^{z_j}$. This is the standard loss for multi-class classification in neural networks.

### Impact of Outliers

MSE penalizes outliers strongly (quadratic growth). MAE is inherently less sensitive (linear). Cross-entropy can be extremely sensitive if the model is overconfident on an outlier. For robustness, consider **Huber loss** (transitions from MSE to MAE for large errors).

### Data Augmentation

Creating new training samples by transforming existing ones to increase dataset diversity and reduce overfitting.

**Geometric transforms:** Random cropping, horizontal/vertical flipping, rotation, scaling, translation.

**Photometric transforms:** Adding noise, blurring, color jittering, grayscale conversion.

**Advanced methods:** Cutout/Random Erasing, Mixup, CutMix.

Important: augmentation must not change the sample's label (e.g., rotating "6" by 180° turns it into "9").

### Handling Class Imbalance

**Oversampling minority class** — Duplicate or augment minority samples (risk: overfitting if naive duplication).

**Undersampling majority class** — Discard majority samples (risk: losing useful data).

**Class weighting in loss** — Assign higher weight to minority class errors (e.g., inverse frequency weighting).

Use appropriate evaluation metrics: per-class accuracy, F1-score, or balanced accuracy instead of raw accuracy.

---

## 12. CNN Architectures (AlexNet, Inception, ResNet), Depth vs. Width, and Batch Normalization

### CNN Principles

CNNs use local receptive fields and **weight sharing**. Filters slide across the image, reducing parameters and providing translation invariance. Pooling subsamples feature maps for translational tolerance and reduced size. Stacking conv + pooling layers builds a hierarchy of features: edges → parts → objects.

### AlexNet (2012)

- 5 convolutional layers + 3 fully-connected layers (8 learnable layers total)
- ~60M parameters
- Key innovations: ReLU activation, dropout in FC layers, GPU training
- First conv: 11×11 with stride 4, 96 kernels
- Max-pooling with overlap (3×3, stride 2)
- Local Response Normalization (LRN)
- Top-5 ImageNet error: ~16% (runner-up was ~26%)

### GoogleNet / Inception v1 (2014)

- 22 layers but only **~5–6.8M parameters** (the paper claims "12× fewer than AlexNet"; detailed layer analysis yields ~6.8M)
- **Inception modules:** parallel convolutions with multiple filter sizes (1×1, 3×3, 5×5) + pooling, concatenated
- 1×1 convolutions as bottlenecks to reduce dimensionality
- Global average pooling instead of large FC layers
- Top-5 accuracy: ~89%

### ResNet (2015)

- 50–152 layers with **residual blocks**: learn $F(x) = H(x) - x$, add input via skip connection
- Skip connections provide gradient highways → solve vanishing gradient problem
- Additional layers can fall back to identity mapping → depth never hurts
- Uses BatchNorm after each conv
- ResNet-152: ~94% top-5 accuracy

### Deeper vs. Wider Networks

Deeper networks represent more complex functions by hierarchical composition — each layer reuses earlier computations. Width increases parallel processing but may learn redundant features. Depth has been the more critical driver of improvements.

### Batch Normalization

Normalizes activations for each mini-batch: subtract batch mean, divide by batch standard deviation, then scale/shift with learnable parameters $\gamma$ and $\beta$. Benefits: allows higher learning rates, acts as mild regularization, mitigates vanishing/exploding gradients, makes networks less sensitive to initialization.

> [!warning] Disputed mechanism
> The original Ioffe & Szegedy (2015) paper claimed BN works by reducing "internal covariate shift." However, Santurkar et al. (NeurIPS 2018) demonstrated that BN's actual benefit is **smoothing the optimization landscape** (reducing the Lipschitz constant of the loss and gradients), making optimization more predictable and stable — not stabilizing input distributions. On an exam, mention both views.

---

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

---

## 14. Semantic Segmentation and U-Net

### Semantic Segmentation

The task of assigning a class label to every pixel in an image. Produces a fine-grained label map (e.g., road, building, person, sky). It is about classes, not instances — two people share the same label "person." Requires both low-level detail (precise boundaries) and high-level context (class identity).

**FCN (Fully Convolutional Network)** — Long et al. (2015) introduced the foundational architecture for semantic segmentation: replace all fully-connected layers in a classification CNN with convolutional layers, enabling pixel-wise output. FCN pioneered the idea of skip connections from earlier layers to improve spatial precision. U-Net builds directly on these FCN concepts.

### U-Net Architecture

An **encoder–decoder** architecture with **skip connections**, shaped like a "U":

- **Encoder (contracting path):** Successive conv layers + pooling reduce spatial resolution while capturing context.
- **Decoder (expansive path):** Upsamples features back to original resolution for pixel-wise predictions.
- **Skip connections:** Feature maps from the encoder are copied to the decoder at corresponding scales, providing high-resolution details lost during pooling.

Each encoder stage: two 3×3 convs + 2×2 pooling. Each decoder stage: upsampling + two 3×3 convs. Final 1×1 conv produces the segmentation map.

### Unpooling, Upsampling, and Transposed Convolution

**Unpooling (Max-Unpooling):** Uses stored max-pooling indices to place values back at original locations, filling others with zeros ("bed of nails" pattern). Not learnable but preserves exact positions of important features. Usually followed by a conv layer.

**Transposed Convolution (Learned Upsampling):** A learnable upsampling layer. Inserts zeros between input positions, then applies a learned convolution filter. Can learn task-specific interpolation. More flexible than fixed interpolation. Risk: checkerboard artifacts if kernel/stride don't align well.

**Upsampling + Convolution:** Simple upsampling (nearest neighbor or bilinear) followed by a standard convolution. Often avoids checkerboard artifacts. Functionally similar to transposed convolution.

---

## 15. 2.5D vs. 3D Imaging, 3D Acquisition Methods, 3D Ultrasound, and Airborne LiDAR

### 2.5D vs. 3D vs. Volumetric 3D

**2.5D** — A single-valued depth function over a 2D grid (e.g., depth map from Kinect). One depth value per pixel. Cannot represent multiple depths along one line of sight, only a single surface from one viewpoint.

**3D (surface)** — Full three-dimensional representation (point cloud, mesh) not tied to a single viewpoint. Can encode multiple surfaces, occluded parts, etc.

**Volumetric 3D** — Data occupying a volume with internal structure (e.g., CT, MRI). A 3D grid of voxels, each with a value (density, tissue type). Captures interior structure, not just surfaces.

### 3D Acquisition Methods

1. **Stereo Vision** — Two cameras, depth via triangulation/disparity.
2. **Structured Light Scanning** — Project known pattern, measure deformation (e.g., Kinect v1).
3. **Time-of-Flight (ToF) / LiDAR** — Measure laser pulse return time for distance.
4. **Photogrammetry (Structure-from-Motion)** — Many photos → camera poses + 3D point cloud.
5. **Laser Scanning (LiDAR)** — Active scanning with high-precision point clouds.
6. **Tomographic Imaging** — CT, MRI, PET → volumetric reconstruction.
7. **Ultrasound Scanning** — Mechanical sweep, 2D matrix array, or freehand with tracker.

### 3D Ultrasound — Factors Influencing Quality

- **Resolution** — Axial > lateral > elevation. Coarse sweep → gaps.
- **Calibration** — Probe-to-tracker calibration accuracy is critical.
- **Speckle and Noise** — Grainy texture inherent to ultrasound.
- **Acoustic Shadowing** — Bone/gas blocks deeper structures.
- **Motion** — Patient/organ movement causes misalignment.
- **Volume Size and Depth** — Trade-off between resolution and coverage.
- **Frame Rate** — Real-time 3D may compromise resolution.
- **User Operation** — Consistent speed and coverage by sonographer.

### Airborne LiDAR and DEM Generation

**Key parameters:** Flight altitude and FOV, laser pulse rate, scan pattern, multiple returns, GPS/INS accuracy, swath overlap, environmental factors.

**DEM Workflow:**
1. Fly with predetermined settings for desired point density/accuracy.
2. Georeference points using GPS/IMU.
3. Clean up (remove outliers, boresight corrections).
4. Classify points (ground vs. non-ground).
5. Interpolate ground points onto a regular grid (TIN, IDW, nearest-neighbor).
6. Output a raster DEM (GeoTIFF, etc.) representing bare-earth terrain.

Typical vertical accuracy: ~5–15 cm RMSE on hard surfaces. DSM (first returns) and DEM (ground) difference gives object heights.

> [!info] DEM vs. DSM vs. nDSM
> **DEM (Digital Elevation Model)** — bare-earth terrain only (buildings and vegetation removed). **DSM (Digital Surface Model)** — includes tops of buildings, trees, and all objects. **nDSM (normalized DSM)** = DSM − DEM, giving the height of objects above ground. LiDAR can produce both since it records multiple returns per pulse.

---

## 16. Single-Camera Calibration, Fundamental Matrix, and Related Concepts

### Single Camera Calibration

**Process:** Take multiple images of a known calibration target (e.g., checkerboard). Detect feature points (corners), establish 2D–3D correspondences, solve for camera parameters (Zhang's method), refine with non-linear optimization (bundle adjustment) minimizing reprojection error.

**Output:**

**Camera Intrinsic Matrix $K$:**

$$K = \begin{pmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

Where $f_x, f_y$ are focal lengths in pixels, $(c_x, c_y)$ is the principal point, $s$ is skew (usually 0).

**Distortion Coefficients:** Radial ($k_1, k_2, k_3$) and tangential ($p_1, p_2$).

**Extrinsics** (per calibration image): Rotation $R$ and translation $t$ of the pattern relative to camera.

### Distortion Types

**Radial distortion** — Barrel (magnification decreases with radius, lines bow outward — wide-angle lenses, $k_1 < 0$) or Pincushion (magnification increases, lines bow inward — telephoto, $k_1 > 0$).

**Tangential distortion** — Lens not perfectly centered/aligned relative to sensor. Causes shearing/stretching. Brown's model: $x_d = x + [2p_1 y + p_2(r^2 + 2x^2)]$, $y_d = y + [p_1(r^2 + 2y^2) + 2p_2 x]$.

### Reprojection Error

Distance (pixels) between observed image points and points projected through the estimated camera model. Good calibration: < 0.5 px RMS.

### Fundamental Matrix

A $3 \times 3$ matrix relating two views: $x'^T F x = 0$. Encodes epipolar geometry — point $x$ in image 1 maps to epipolar line $l' = Fx$ in image 2.

**Decomposition:** Essential matrix $E = K'^T F K$ relates purely to extrinsics: $E = [t]_\times R$. SVD of $E$ yields rotation $R$ and translation $t$ (up to scale). Four possible solutions; choose the one with positive depth (cheirality constraint).

> [!tip] Key properties and algorithms for exams
> The fundamental matrix $F$ is **rank 2** ($\det(F) = 0$) and has **7 degrees of freedom**. The essential matrix $E$ has **5 DOF**. The **8-point algorithm** estimates $F$ from ≥8 point correspondences by solving a linear system. In practice, **RANSAC** (Random Sample Consensus) is used alongside the 8-point algorithm to handle outliers: randomly sample minimal sets of correspondences, compute $F$ candidates, and select the one with most inliers.

### Homogeneous Coordinates

Represent 2D points as 3-vectors $(x, y, w) \rightarrow (x/w, y/w)$. Purpose:
- Unify translation and rotation into matrix multiplications
- Represent perspective projection linearly
- Handle points at infinity (e.g., vanishing points) when $w = 0$
- Enable elegant formulas (line through two points = cross product of homogeneous vectors)

---

## 17. Stereo Camera Calibration, Rectification, and 3D from Stereo

### Stereo Calibration

Determines relative pose ($R$, $T$) between two cameras using a calibration object visible in both cameras simultaneously. Output: rotation $R$ (3×3), translation $T$ (3×1), essential matrix $E$, fundamental matrix $F$.

### Image Rectification

Warps both images so that epipolar lines become horizontal and aligned on the same scanlines. Result: corresponding points have the **same y-coordinate**, reducing correspondence search to 1D along rows.

### Correspondence Problem

For each pixel in the left image, find the matching pixel in the right image. Challenges: textureless regions, repeated patterns, occlusions, perspective distortion, radiometric differences.

**Block matching:** For each pixel, slide a window along the same row in the right image, compute similarity (SSD, NCC), find minimum cost. Enforce uniqueness constraint and left-right consistency check.

### Triangulation and Disparity

With known camera poses and matched correspondences, compute 3D via triangulation (intersecting rays from both cameras).

**Disparity-depth relation:**

$$Z = \frac{f \cdot B}{d}$$

Where $f$ is focal length, $B$ is baseline, $d$ is disparity ($x_{\text{left}} - x_{\text{right}}$). Large disparity → close object. Small disparity → far object.

**Depth error propagation:** $\delta Z \approx \frac{Z^2}{f \cdot B} \cdot \delta d$ — depth error grows **quadratically** with distance for a given disparity error $\delta d$. This is why stereo is much more accurate for near objects than far ones. To improve far-depth accuracy, increase baseline $B$ or camera resolution.

### Problematic Cases

- **Occlusion** — Points visible in one camera only. Detected via left-right consistency checks.
- **Textureless areas** — Flat matching cost → ambiguous disparity.
- **Repetitive patterns** — Multiple similar-cost matches (e.g., picket fence).
- **Thin/transparent objects** — Aliasing or background seen through.
- **Radiometric differences** — Different exposure/gain between cameras.
- **Calibration errors** — Imprecise rectification.
- **Boundary effects** — Block covers foreground + background → edge fattening.

### Subpixel Disparity

Fit a parabola to cost values at $(d-1, d, d+1)$ and find the analytical minimum for fractional pixel precision (e.g., 1/16th pixel).

> [!tip] Semi-Global Matching (SGM)
> The dominant modern stereo method. Instead of matching blocks independently, SGM aggregates matching costs along multiple paths (typically 8 or 16 directions) across the image, enforcing smoothness penalties for disparity changes. This dramatically improves results in textureless areas and at depth discontinuities while remaining computationally efficient.

---

## 18. Structure-from-Motion (SfM), Multi-View Stereo (MVS), and 3D Reconstruction

### Structure-from-Motion (SfM)

Recovers 3D structure and camera poses from multiple images of a static scene. Pipeline:

1. Detect features (e.g., SIFT) in all images.
2. Match features between image pairs.
3. Start with two images: compute essential matrix, decompose to get relative pose, triangulate initial points.
4. Incrementally add images: use PnP to solve pose from 3D–2D correspondences.
5. Triangulate new points.
6. Run **bundle adjustment** to jointly refine all camera poses and 3D points, minimizing reprojection error.

Output: sparse 3D point cloud + camera poses.

### Multi-View Stereo (MVS)

Uses calibrated cameras from SfM to compute **dense** depth maps or point clouds. Approaches: patch-based MVS (CMVS/PMVS), plane-sweep algorithms, depth-from-view clusters. Fuses multiple depth maps into a consistent dense reconstruction.

### Full 3D Reconstruction Pipeline

1. Feature detection & matching
2. SfM → camera poses + sparse point cloud
3. Bundle Adjustment
4. MVS → dense points / depth maps
5. Depth Map Fusion / Surface Reconstruction (TSDF, Poisson reconstruction)
6. Texture mapping

### Bundle Adjustment

Joint non-linear refinement of structure (3D points) and camera parameters to minimize reprojection error: $\sum_{i,j} |x_{ij}^{\text{obs}} - \pi(P_i, X_j)|^2$. Uses Levenberg-Marquardt with sparse Jacobian structure. The "gold standard" refinement method for multi-view geometry.

### Representations

**Point Cloud** — Set of 3D points (+ color). Direct output of SfM/MVS. Easy to work with (ICP registration, PointNet). No explicit surface connectivity.

**Voxel Grid** — Partitions space into cubic cells. Good for volumetric operations, 3D CNNs, occupancy queries (robotics). Memory-intensive ($O(n^3)$); mitigated with octrees.

**Polygon Mesh** — Vertices, edges, faces. Standard for graphics/CAD. Compact surface representation, easy to render. Needed for 3D printing, simulation, AR/VR.

---

## 19. 3D Feature Detection, 3D Convolution, and Octree Networks

### Feature Detection in 3D

**3D Harris** — On 3D surfaces/point clouds, detect points with high curvature in multiple directions (analogous to 2D corner detection using surface normals/curvatures).

**3D SIFT** — Extended to space-time volumes (video) or spatial 3D data. Build 3D scale-space (Gaussian smoothing with increasing σ), detect DoG extrema in 3D, compute descriptors from 3D gradient distributions.

**Space-time features (video)** — Harris3D by Laptev: compute spatial and temporal gradients, find points with large variations in both space and time. Useful for action recognition.

Other 3D descriptors: SHOT, FPFH, PFH, ISS.

### 2D vs. 3D Convolution

**2D convolution:** Kernel slides across 2D (width, height). Processes spatial features.

**3D convolution:** Kernel is 3D (width, height, depth/time), slides in three directions. Output is a 3D activation map. Captures context in the extra dimension (time for video, spatial depth for volumetric data). More parameters and computationally heavier than 2D.

Applications: video classification (action recognition), medical imaging (volumetric segmentation), 3D shape classification.

### OctNet (Octree-based 3D CNN)

**Problem:** Dense voxel grids waste memory on empty space — most 3D scenes are mostly empty.

**Solution:** OctNet (Riegler et al., CVPR 2017) uses a **hybrid grid-octree structure** where empty or uniform regions are represented at coarser octree levels, while detailed regions (near surfaces) are subdivided to finer voxels. The octree recursively partitions space, and OctNet modifies convolution operations to traverse neighbor nodes in this tree structure. This is distinct from O-CNN (Wang et al., 2017), which restricts convolution strictly to surface-occupied octants.

**Benefits:** Huge reduction in memory and computation for sparse 3D data. Can handle resolutions like 256³ that would be impossible as dense grids. Suitable for 3D object classification and semantic segmentation.

**Alternatives for 3D recognition:**
- Multi-view approach — Render 2D views, use 2D CNN ensemble (often best classification accuracy)
- Voxelize + 3D CNN (with or without octree)
- **PointNet/PointNet++** (Qi et al., 2017) — Directly operate on raw point sets using shared MLPs and a symmetric aggregation function (max pooling) to achieve permutation invariance. PointNet++ adds hierarchical grouping to capture local structure. Currently state-of-the-art for many point cloud segmentation tasks (e.g., LiDAR scene segmentation in autonomous driving).

Each approach trades off between maturity of 2D CNNs, handling of high-resolution 3D detail, and memory efficiency.
