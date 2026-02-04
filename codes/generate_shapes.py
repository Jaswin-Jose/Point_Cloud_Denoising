"""
I created this file entirely using claude sonnet 4.5. This file generates the following shapes. 

Shapes include:
- Hyperboloid
- Cylinder
- Cone
- Torus (Donut)
- Star
- Chair (simplified model)
- Sphere with bumps
"""

import numpy as np
from typing import Tuple


class ShapeGenerator:
    def __init__(self, num_points: int = 3200, noise_std: float = 1.0):
        self.num_points = num_points
        self.noise_std = noise_std
    
    def add_noise(self, points: np.ndarray) -> np.ndarray:
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        avg_bbox = np.mean(bbox_size)
        
        actual_noise_std = self.noise_std * 0.01 * avg_bbox
        
        noise = np.random.normal(0, actual_noise_std, points.shape)
        return points + noise
    
    def generate_hyperboloid(self) -> Tuple[np.ndarray, np.ndarray]:
        u = np.random.uniform(0, 2*np.pi, self.num_points)
        v = np.random.uniform(-2, 2, self.num_points)
        
        a, b, c = 1.0, 1.0, 1.0
        x = a * np.cosh(v) * np.cos(u)
        y = b * np.cosh(v) * np.sin(u)
        z = c * np.sinh(v)
        
        clean_points = np.column_stack([x, y, z])
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_cylinder(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.uniform(0, 2*np.pi, self.num_points)
        z = np.random.uniform(-2, 2, self.num_points)
        radius = 1.0
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        clean_points = np.column_stack([x, y, z])
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_cone(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.uniform(0, 2*np.pi, self.num_points)
        z = np.random.uniform(0, 2, self.num_points)
        radius = 1.0 - 0.5 * z  
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        clean_points = np.column_stack([x, y, z])
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_torus(self) -> Tuple[np.ndarray, np.ndarray]:
        u = np.random.uniform(0, 2*np.pi, self.num_points)
        v = np.random.uniform(0, 2*np.pi, self.num_points)
        
        R = 2.0  
        r = 0.5  
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        clean_points = np.column_stack([x, y, z])
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_star(self) -> Tuple[np.ndarray, np.ndarray]:
        points_per_part = self.num_points // 6
        all_points = []
        
        n_center = points_per_part
        theta = np.random.uniform(0, 2*np.pi, n_center)
        phi = np.random.uniform(0, np.pi, n_center)
        r_center = 0.3
        
        x = r_center * np.sin(phi) * np.cos(theta)
        y = r_center * np.sin(phi) * np.sin(theta)
        z = r_center * np.cos(phi)
        all_points.append(np.column_stack([x, y, z]))
        
        n_points = 5
        for i in range(n_points):
            angle = i * 2 * np.pi / n_points
            
            t = np.random.uniform(0, 1, points_per_part)
            radius = 0.3 * (1 - 0.8 * t)
            length = 1.5 * t
            
            theta_ray = np.random.uniform(0, 2*np.pi, points_per_part)
            
            x_ray = length * np.cos(angle) + radius * np.cos(theta_ray) * np.sin(angle)
            y_ray = length * np.sin(angle) + radius * np.cos(theta_ray) * np.cos(angle)
            z_ray = radius * np.sin(theta_ray)
            
            all_points.append(np.column_stack([x_ray, y_ray, z_ray]))
        
        clean_points = np.vstack(all_points)[:self.num_points]
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_chair(self) -> Tuple[np.ndarray, np.ndarray]:
        points_per_part = self.num_points // 6
        all_points = []
        
        x_seat = np.random.uniform(-1, 1, points_per_part)
        y_seat = np.random.uniform(-1, 1, points_per_part)
        z_seat = np.ones(points_per_part) * 1.0
        all_points.append(np.column_stack([x_seat, y_seat, z_seat]))
        
        x_back = np.random.uniform(-1, 1, points_per_part)
        y_back = np.ones(points_per_part) * 1.0
        z_back = np.random.uniform(1.0, 2.5, points_per_part)
        all_points.append(np.column_stack([x_back, y_back, z_back]))
        
        leg_positions = [(-0.8, -0.8), (0.8, -0.8), (-0.8, 0.8), (0.8, 0.8)]
        for leg_x, leg_y in leg_positions:
            x_leg = np.random.normal(leg_x, 0.1, points_per_part)
            y_leg = np.random.normal(leg_y, 0.1, points_per_part)
            z_leg = np.random.uniform(0, 1.0, points_per_part)
            all_points.append(np.column_stack([x_leg, y_leg, z_leg]))
        
        clean_points = np.vstack(all_points)[:self.num_points]
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_bumpy_sphere(self) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.uniform(0, 2*np.pi, self.num_points)
        phi = np.random.uniform(0, np.pi, self.num_points)
        
        r = 1.0 + 0.2 * (np.sin(3*theta) * np.cos(2*phi) + 
                         np.cos(4*theta) * np.sin(3*phi))
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        clean_points = np.column_stack([x, y, z])
        noisy_points = self.add_noise(clean_points)
        
        return clean_points, noisy_points
    
    def generate_all_shapes(self) -> dict:
        shapes = {
            'hyperboloid': self.generate_hyperboloid(),
            'cylinder': self.generate_cylinder(),
            'cone': self.generate_cone(),
            'torus': self.generate_torus(),
            'star': self.generate_star(),
            'chair': self.generate_chair(),
            'bumpy_sphere': self.generate_bumpy_sphere()
        }
        return shapes


def save_point_cloud_ply(points: np.ndarray, filename: str):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")