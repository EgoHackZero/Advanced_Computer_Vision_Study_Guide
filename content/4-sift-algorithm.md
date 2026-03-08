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