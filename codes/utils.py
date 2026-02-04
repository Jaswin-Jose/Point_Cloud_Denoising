import numpy as np
import matplotlib.pyplot as plt
import os
import csv

def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nbrs.fit(points2)
    distances1, _ = nbrs.kneighbors(points1)
    nbrs.fit(points1)
    distances2, _ = nbrs.kneighbors(points2)
    cd = np.mean(distances1) + np.mean(distances2)
    
    return cd

def mean_squared_error(points1: np.ndarray, points2: np.ndarray) -> float:
    if len(points1) != len(points2):
        return chamfer_distance(points1, points2)
    
    return np.mean(np.sum((points1 - points2)**2, axis=1))


def visualize_point_cloud(clean: np.ndarray, noisy: np.ndarray, 
                         denoised: np.ndarray, title: str,
                         save_path: str = None):
    fig = plt.figure(figsize=(15, 5))
    
    is_star = 'star' in title.lower()
    elevation = 90 if is_star else 30
    azimuth = 0 if is_star else 45
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(clean[:, 0], clean[:, 1], clean[:, 2], 
               c='blue', marker='.', s=1, alpha=0.6)
    ax1.set_title('Clean')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=elevation, azim=azimuth)
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(noisy[:, 0], noisy[:, 1], noisy[:, 2], 
               c='red', marker='.', s=1, alpha=0.6)
    ax2.set_title('Noisy')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=elevation, azim=azimuth)
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(denoised[:, 0], denoised[:, 1], denoised[:, 2], 
               c='green', marker='.', s=1, alpha=0.6)
    ax3.set_title('Denoised')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=elevation, azim=azimuth)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()


def save_statistics(stats: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=stats.keys())
        writer.writeheader()
        writer.writerow(stats)
