# Point Cloud Denoising Project

This project implements and evaluates various point cloud denoising algorithms on synthetic 3D shapes with controlled Gaussian noise.

## Project Structure

```
PCD/
├── codes/
│   ├── generate_shapes.py                              # Shape generation
│   ├── algorithm1_robust_statistical_normal_filtering.py
│   ├── algorithm2_pointnet.py
│   ├── algorithm3_bilateral_filtering.py
│   ├── algorithm4_moving_least_squares.py
│   ├── algorithm5_statistical_outlier_removal.py
│   ├── utils.py                                        # Utilities and metrics
│   └── main.py                                         # Main execution script
├── images/                                             # Visualization outputs
│   ├── hyperboloid/
│   ├── cylinder/
│   ├── cone/
│   ├── torus/
│   ├── star/
│   ├── chair/
│   └── bumpy_sphere/
├── stats/                                              # CSV statistics
│   ├── hyperboloid/
│   ├── cylinder/
│   ├── cone/
│   ├── torus/
│   ├── star/
│   ├── chair/
│   ├── bumpy_sphere/
│   └── summary_report.csv
└── point_clouds/                                       # PLY files
    ├── hyperboloid/
    ├── cylinder/
    ├── cone/
    ├── torus/
    ├── star/
    ├── chair/
    └── bumpy_sphere/
```

## Denoising Algorithms

### 1. Robust Statistical Normal Filtering
**Reference**: Yadav et al. (2020), "Robust Normal Filtering of Point Clouds via Statistical Neighborhood Analysis"

Uses robust statistics to estimate normals and filter noise based on normal consistency.

### 2. PointNet
**Reference**: Qi et al. (2017), "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"

Deep learning-based approach using PointNet autoencoder architecture for denoising.

### 3. Bilateral Filtering
**Reference**: Fleishman et al. (2003), "Bilateral Mesh Denoising" and Digne (2012), "Similarity Based Filtering of Point Clouds"

Preserves edges while smoothing by considering both spatial proximity and feature similarity.

### 4. Moving Least Squares (MLS)
**Reference**: Alexa et al. (2003), "Computing and Rendering Point Set Surfaces" 
### 5. Statistical Outlier Removal (SOR)
**Reference**: Rusu and Cousins (2011), "3D is here: Point Cloud Library (PCL)"

### Run All Experiments
```bash
cd codes
python main.py
```
