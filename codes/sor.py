import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


class StatisticalOutlierRemoval:
    def __init__(self, k_neighbors: int = 30, std_mul: float = 1.0):
        self.k_neighbors = k_neighbors
        self.std_mul = std_mul
    
    def compute_mean_distances(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='kd_tree')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Exclude self (first neighbor is always the point itself)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Compute mean distance for each point
        mean_distances = np.mean(distances, axis=1)
        
        return mean_distances, indices
    
    def identify_outliers(self, mean_distances: np.ndarray) -> np.ndarray:
        # Compute global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        # Threshold
        threshold = global_mean + self.std_mul * global_std
        
        # Identify outliers
        is_outlier = mean_distances > threshold
        
        return is_outlier
    
    def correct_outliers(self,
                        points: np.ndarray,
                        is_outlier: np.ndarray,
                        indices: np.ndarray) -> np.ndarray:
        # Keep only inlier points
        inlier_points = points[~is_outlier]
        return inlier_points

    
    def denoise(self, points: np.ndarray) -> np.ndarray:
        # Compute mean distances
        mean_distances, indices = self.compute_mean_distances(points)
        
        # Identify outliers
        is_outlier = self.identify_outliers(mean_distances)
        
        # Correct outliers
        denoised_points = self.correct_outliers(points, is_outlier, indices)
        
        return denoised_points

def denoise_point_cloud(noisy_points: np.ndarray,
                        k_neighbors: int = 30,
                        std_mul: float = 1.0) -> np.ndarray:
    sor = StatisticalOutlierRemoval(k_neighbors, std_mul)
    return sor.denoise(noisy_points)


if __name__ == "__main__":
    import os
    from generate_shapes import ShapeGenerator, save_point_cloud_ply
    from utils import chamfer_distance, mean_squared_error, visualize_point_cloud, save_statistics
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    generator = ShapeGenerator(num_points=37000, noise_std=1.5)
    shapes = generator.generate_all_shapes()
    
    shape_names = list(shapes.keys())
    
    algo_name = "Statistical_Outlier_Removal"
    
    for shape_name, (clean, noisy) in shapes.items():
        print(shape_name.upper())
        os.makedirs(os.path.join(base_path, 'images', shape_name), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'stats', shape_name), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'point_clouds', shape_name), exist_ok=True)
        
        denoised = denoise_point_cloud(noisy)
        
        cd = chamfer_distance(clean, denoised)
        mse = mean_squared_error(clean, denoised)
        
        print(f"Chamfer Distance: {cd:.6f}")
        print(f"MSE: {mse:.6f}")
        
        denoised_path = os.path.join(base_path, 'point_clouds', shape_name, f'{algo_name}.ply')
        save_point_cloud_ply(denoised, denoised_path)
        
        img_path = os.path.join(base_path, 'images', shape_name, f'{algo_name}.png')
        visualize_point_cloud(clean, noisy, denoised, f'{shape_name} - {algo_name}', img_path)
        stats = {
            'shape': shape_name,
            'algorithm': algo_name,
            'chamfer_distance': cd,
            'mse': mse,
            'num_points': len(clean)
        }
        stats_path = os.path.join(base_path, 'stats', shape_name, f'{algo_name}.csv')
        save_statistics(stats, stats_path)