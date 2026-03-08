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