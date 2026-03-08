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