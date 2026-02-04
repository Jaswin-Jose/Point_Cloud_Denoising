import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class LBFNet(nn.Module):
    def __init__(self, num_scales=3, num_points=64):
        super(LBFNet, self).__init__()
        self.num_scales = num_scales
        self.num_points = num_points
        
        self.encoder = nn.Module()
        self.encoder.mlp = nn.Sequential(
            nn.Linear(3, 64),          
            nn.ReLU(),                  
            nn.Linear(64, 128),         
            nn.ReLU(),                   
            nn.Linear(128, 256),        
            nn.ReLU(),                   
            nn.Linear(256, 512),        
            nn.ReLU()                    
        )
        
        self.decoder = nn.Module()
        self.decoder.fc = nn.Sequential(
            nn.Linear(512, 128),       
            nn.ReLU(),                   
            nn.Linear(128, 2)           
        )
        
    def forward(self, x):
        
        x = x.mean(dim=1).mean(dim=1)  
        
        features = self.encoder.mlp(x)  
        
        params = self.decoder.fc(features) 
        
        sigma_d = F.softplus(params[:, 0])
        sigma_n = F.softplus(params[:, 1])
        
        return sigma_d, sigma_n


class LBFDenoiser:
    def __init__(self, model_path, device='cuda', num_scales=3, radii=[0.05, 0.1, 0.15]):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_scales = num_scales
        self.radii = radii
        
        self.model = LBFNet(num_scales=num_scales)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded LBF model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Using device: {self.device}")
    
    def estimate_normals(self, points, k=20):
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
        nbrs.fit(points)
        _, indices = nbrs.kneighbors(points)
        
        normals = np.zeros_like(points)
        for i in range(len(points)):
            neighbors = points[indices[i]]
            
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]
        
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        return normals
    
    def get_multiscale_patches(self, points, center_idx, radii):
        center = points[center_idx]
        patches = []
        
        for r in radii:
            distances = np.linalg.norm(points - center, axis=1)
            mask = distances < r
            neighbors = points[mask]
            
            if len(neighbors) > 3:
                pca = PCA(n_components=3)
                normalized = pca.fit_transform(neighbors - center)
                
                if len(normalized) > 64:
                    indices = np.random.choice(len(normalized), 64, replace=False)
                    patch = normalized[indices]
                else:
                    patch = np.zeros((64, 3))
                    patch[:len(normalized)] = normalized
            else:
                patch = np.zeros((64, 3))
            
            patches.append(patch)
        
        return np.array(patches)  
    
    def bilateral_filter(self, points, normals, center_idx, sigma_d, sigma_n, radius=0.1):
        center = points[center_idx]
        normal = normals[center_idx]
        
        distances = np.linalg.norm(points - center, axis=1)
        mask = distances < radius
        neighbors = points[mask]
        
        if len(neighbors) < 2:
            return 0.0
        
        relative = neighbors - center
        
        spatial_dist = np.linalg.norm(relative, axis=1)
        
        projections = np.dot(relative, normal)
        
        w_d = np.exp(-spatial_dist**2 / (2 * sigma_d**2 + 1e-8))
        w_n = np.exp(-np.abs(projections)**2 / (2 * sigma_n**2 + 1e-8))
        
        weights = w_d * w_n
        
        numerator = np.sum(weights * projections)
        denominator = np.sum(weights) + 1e-8
        
        delta = numerator / denominator
        
        return delta
    
    def denoise(self, noisy_points, batch_size=32, num_iterations=1):
        points = noisy_points.copy()
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            normals = self.estimate_normals(points)
            
            displacements = np.zeros(len(points))
            
            for i in range(0, len(points), batch_size):
                batch_end = min(i + batch_size, len(points))
                batch_indices = range(i, batch_end)
                
                batch_patches = []
                for idx in batch_indices:
                    patches = self.get_multiscale_patches(points, idx, self.radii)
                    batch_patches.append(patches)
                
                batch_patches = np.array(batch_patches)  
                
                with torch.no_grad():
                    patches_tensor = torch.FloatTensor(batch_patches).to(self.device)
                    sigma_d_batch, sigma_n_batch = self.model(patches_tensor)
                    sigma_d_batch = sigma_d_batch.cpu().numpy()
                    sigma_n_batch = sigma_n_batch.cpu().numpy()
                
                for j, idx in enumerate(batch_indices):
                    delta = self.bilateral_filter(
                        points, normals, idx,
                        sigma_d_batch[j], sigma_n_batch[j],
                        radius=max(self.radii)
                    )
                    displacements[idx] = delta
                
                if (i + batch_size) % 1000 == 0:
                    print(f"  Processed {min(i + batch_size, len(points))}/{len(points)} points")
            
            points = points + displacements.reshape(-1, 1) * normals
        
        return points

if __name__ == "__main__":
    import os
    from generate_shapes import ShapeGenerator, save_point_cloud_ply
    from utils import chamfer_distance, mean_squared_error, visualize_point_cloud, save_statistics

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(base_dir, 'model', 'lbf_best.pth')
    
    generator = ShapeGenerator(num_points=37000, noise_std=1.5)
    shapes = generator.generate_all_shapes()
    
    shape_names = list(shapes.keys())
    
    algo_name = "Learned Bilateral Filtering"
    params = {'batch_size': 32, 'num_iterations': 3}
    
    
    for shape_name, (clean, noisy) in shapes.items():
        print(shape_name.upper())
        os.makedirs(os.path.join(base_dir, 'images', shape_name), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'stats', shape_name), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'point_clouds', shape_name), exist_ok=True)
        

        denoiser = LBFDenoiser(
            model_path=model,
            device=device
        )

        denoised = denoiser.denoise(noisy, **params)
        
        cd = chamfer_distance(clean, denoised)
        mse = mean_squared_error(clean, denoised)
        
        print(f"Chamfer Distance: {cd:.6f}")
        print(f"MSE: {mse:.6f}")
        
        denoised_path = os.path.join(base_dir, 'point_clouds', shape_name, f'{algo_name}.ply')
        save_point_cloud_ply(denoised, denoised_path)
        
        img_path = os.path.join(base_dir, 'images', shape_name, f'{algo_name}.png')
        visualize_point_cloud(clean, noisy, denoised, f'{shape_name} - {algo_name}', img_path)
        stats = {
            'shape': shape_name,
            'algorithm': algo_name,
            'chamfer_distance': cd,
            'mse': mse,
            'num_points': len(clean)
        }
        stats_path = os.path.join(base_dir, 'stats', shape_name, f'{algo_name}.csv')
        save_statistics(stats, stats_path)
    
