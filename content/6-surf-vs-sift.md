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