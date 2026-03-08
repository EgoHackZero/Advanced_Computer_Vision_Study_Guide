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