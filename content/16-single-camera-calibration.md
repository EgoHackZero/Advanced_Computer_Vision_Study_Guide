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