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