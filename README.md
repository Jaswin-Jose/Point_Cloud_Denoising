# Point Cloud Denoising Project

This project implements and evaluates various point cloud denoising algorithms on synthetic 3D shapes with controlled Gaussian noise.

## Project Structure

```
PCD/
├── codes/
│   ├── generate_shapes.py                              # Shape generation
│   ├── lbf.py
│   ├── mls.py
│   ├── sor.py
│   ├── robust.py                                      
│   └── utils.py                                        
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

### 2. Learning Based Bilateral Filtering
**Reference**: Huajian et al. (2022), "LBF: Learnable Bilateral Filter for Point Cloud Denoising"

### 3. Moving Least Squares (MLS)
**Reference**: Alexa et al. (2003), "Computing and Rendering Point Set Surfaces" 

### 4. Statistical Outlier Removal (SOR)
**Reference**: Rusu and Cousins (2011), "3D is here: Point Cloud Library (PCL)"

### Run All Experiments
```bash
cd codes
python main.py
```
### To compare
```bash
python3 generate_comparisons.py
```
