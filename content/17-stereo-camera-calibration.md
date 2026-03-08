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