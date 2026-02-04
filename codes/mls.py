import numpy as np
from sklearn.neighbors import NearestNeighbors

class AlexaMLS:
    def __init__(self, h=0.5, poly_degree=3, max_iter=10, tol=1e-6):
        self.h = h
        self.poly_degree = poly_degree
        self.max_iter = max_iter
        self.tol = tol

    def weight(self, d):
        return np.exp(-(d * d) / (self.h * self.h))

    # ------------------------------------------------------------
    # Step 1: compute reference plane
    # ------------------------------------------------------------
    def compute_reference_plane(self, r, neighbors):
        n = np.array([0.0, 0.0, 1.0])  # initial normal
        t = 0.0

        for _ in range(self.max_iter):
            q = r + t * n
            diffs = neighbors - q
            dists = np.linalg.norm(diffs, axis=1)
            w = self.weight(dists)

            # Weighted covariance matrix B 
            B = np.zeros((3, 3))
            for i in range(len(neighbors)):
                B += w[i] * np.outer(diffs[i], diffs[i])

            # Update normal: smallest eigenvalue eigenvector
            _, eigvecs = np.linalg.eigh(B)
            n_new = eigvecs[:, 0]
            n_new /= np.linalg.norm(n_new)

            # Minimize in t (1D): project r onto new normal
            t_new = np.dot(n_new, (neighbors - r).mean(axis=0))

            if abs(t_new - t) < self.tol:
                break

            n, t = n_new, t_new

        return n, t, r + t * n

    def fit_polynomial(self, q, n, neighbors):
        # Build local frame
        u = np.cross(n, [1, 0, 0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(n, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        X, Y, F, W = [], [], [], []

        for p in neighbors:
            diff = p - q
            x = np.dot(diff, u)
            y = np.dot(diff, v)
            f = np.dot(diff, n)
            d = np.linalg.norm(diff)

            X.append(x)
            Y.append(y)
            F.append(f)
            W.append(self.weight(d))

        X, Y, F, W = map(np.array, (X, Y, F, W))

        # Polynomial basis
        terms = []
        for i in range(self.poly_degree + 1):
            for j in range(self.poly_degree + 1 - i):
                terms.append((i, j))

        A = np.zeros((len(X), len(terms)))
        for k, (i, j) in enumerate(terms):
            A[:, k] = (X ** i) * (Y ** j)

        Wmat = np.diag(W)
        coeffs = np.linalg.lstsq(Wmat @ A, Wmat @ F, rcond=None)[0]

        # g(0,0) is constant term
        return coeffs[0]

    # ------------------------------------------------------------
    # MLS projection operator P(r)
    # ------------------------------------------------------------
    def project(self, r, neighbors):
        n, t, q = self.compute_reference_plane(r, neighbors)
        g00 = self.fit_polynomial(q, n, neighbors)
        return q + g00 * n

def denoise_point_cloud(points, h=0.5, k=30, iterations=1):
    mls = AlexaMLS(h=h)
    points_current = points.copy()

    for _ in range(iterations):
        nbrs = NearestNeighbors(n_neighbors=k).fit(points_current)
        _, indices = nbrs.kneighbors(points_current)

        new_points = np.zeros_like(points_current)

        for i, p in enumerate(points_current):
            neighbors = points_current[indices[i]]
            new_points[i] = mls.project(p, neighbors)

        points_current = new_points

    return points_current


if __name__ == "__main__":
    import os
    from generate_shapes import ShapeGenerator, save_point_cloud_ply
    from utils import chamfer_distance, mean_squared_error, visualize_point_cloud, save_statistics
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    generator = ShapeGenerator(num_points=37000, noise_std=1.5)
    shapes = generator.generate_all_shapes()
    
    shape_names = list(shapes.keys())
    
    algo_name = "Moving_Least_Squares"
    params = {'k': 30, 'iterations': 3, 'h': 0.2}
    
    
    for shape_name, (clean, noisy) in shapes.items():
        print(shape_name.upper())
        print("MLS takes a lot of time. So, you may have to wait or decrease the number of points from 37000 to 3200 or 2048 in line 121")
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
    
