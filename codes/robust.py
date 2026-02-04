import numpy as np
from sklearn.neighbors import NearestNeighbors

def gaussian_weight(x, sigma):
    return np.exp(-(x ** 2) / (sigma ** 2))

def tukey_weight(x, sigma):
    w = np.zeros_like(x)
    mask = np.abs(x) <= sigma
    w[mask] = (1 - (x[mask] / sigma) ** 2) ** 2
    return w

def estimate_normals_pca(points, k_neighbors=30):
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighbors = points[indices[i]]
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid

        cov = centered.T @ centered / len(neighbors)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        normals[i] = eigenvectors[:, 0]  

    return normals

def bilateral_normal_filter(
    points,
    normals,
    k_neighbors=30,
    sigma=0.3,
    sigma_d=0.1
):
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    _, indices = nbrs.kneighbors(points)

    filtered_normals = np.zeros_like(normals)

    for i in range(len(points)):
        ni = normals[i]
        pi = points[i]

        nj = normals[indices[i]]
        pj = points[indices[i]]

        x = np.linalg.norm(nj - ni, axis=1)
        d = np.linalg.norm(pj - pi, axis=1)
        g = gaussian_weight(x, sigma)
        f = np.exp(-(d ** 2) / (sigma_d ** 2))

        w = g * f

        n_tilde = np.sum(w[:, None] * nj, axis=0)

        norm = np.linalg.norm(n_tilde)
        if norm > 0:
            filtered_normals[i] = n_tilde / norm
        else:
            filtered_normals[i] = ni

    return filtered_normals

def project_points_to_normals(points, normals, k_neighbors=30, step=0.1):

    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    _, indices = nbrs.kneighbors(points)

    new_points = points.copy()

    for i in range(len(points)):
        pi = points[i]
        ni = normals[i]

        pj = points[indices[i]]
        centroid = pj.mean(axis=0)

        displacement = np.dot(centroid - pi, ni) * ni
        new_points[i] = pi + step * displacement

    return new_points

def denoise_point_cloud(
    points,
    iterations=5,
    k_neighbors=30,
    sigma=0.3,
    sigma_d=0.1,
    step=0.1
):
    normals = estimate_normals_pca(points)
    P = points.copy()
    N = normals.copy()

    for _ in range(iterations):
        N = bilateral_normal_filter(
            P, N,
            k_neighbors=k_neighbors,
            sigma=sigma,
            sigma_d=sigma_d
        )
        P = project_points_to_normals(
            P, N,
            k_neighbors=k_neighbors,
            step=step
        )

    return P


if __name__ == "__main__":
    import os
    from generate_shapes import ShapeGenerator, save_point_cloud_ply
    from utils import chamfer_distance, mean_squared_error, visualize_point_cloud, save_statistics

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    generator = ShapeGenerator(num_points=37000, noise_std=1.5)
    shapes = generator.generate_all_shapes()
    
    shape_names = list(shapes.keys())
    
    algo_name = "Robust_Statistical_Normal_Filtering"
    params = {'k_neighbors': 30, 'sigma': 0.3, 'sigma_d': 0.1, 'iterations': 5, 'step': 0.1}
    
    for shape_name, (clean, noisy) in shapes.items():
        
        os.makedirs(os.path.join(base_path, 'images', shape_name), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'stats', shape_name), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'point_clouds', shape_name), exist_ok=True)
        
        denoised = denoise_point_cloud(noisy, **params)
        
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
    
    print("Processing complete!")