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