import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv


def generate_summary_report(base_path: str = '/home/jaswinjose/Desktop/PCD'):
    stats_dir = os.path.join(base_path, 'stats')
    output_path = os.path.join(stats_dir, 'summary_report.csv')
    
    all_stats = []
    
    for shape_name in os.listdir(stats_dir):
        shape_dir = os.path.join(stats_dir, shape_name)
        if not os.path.isdir(shape_dir):
            continue
        for csv_file in os.listdir(shape_dir):
            if not csv_file.endswith('.csv'):
                continue
            
            csv_path = os.path.join(shape_dir, csv_file)
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_stats.append(row)
    
    if all_stats:
        fieldnames = all_stats[0].keys()
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
        
        print(f"Report generated")
    else:
        print("No statistics found to aggregate!")


def load_summary_stats(stats_path: str) -> dict:
    stats = {}
    
    if not os.path.exists(stats_path):
        return stats
    
    with open(stats_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            shape = row['shape']
            if shape not in stats:
                stats[shape] = {}
            
            algo = row['algorithm']
            stats[shape][algo] = {
                'chamfer': float(row['chamfer_distance']),
                'mse': float(row['mse'])
            }
    
    return stats


def plot_algorithm_comparison(stats_path: str, output_path: str):
    
    stats = load_summary_stats(stats_path)
    
    if not stats:
        print("No statistics found!")
        return
    
    shapes = list(stats.keys())
    algorithms = list(stats[shapes[0]].keys()) if shapes else []
    
    n_shapes = len(shapes)
    n_algos = len(algorithms)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    x = np.arange(n_shapes)
    width = 0.15
    
    ax = axes[0]
    for i, algo in enumerate(algorithms):
        chamfer_values = [stats[shape][algo]['chamfer'] for shape in shapes]
        offset = (i - n_algos/2) * width
        ax.bar(x + offset, chamfer_values, width, label=algo)
    
    ax.set_xlabel('Shapes', fontsize=12)
    ax.set_ylabel('Chamfer Distance', fontsize=12)
    ax.set_title('Chamfer Distance Comparison Across Algorithms', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1]
    for i, algo in enumerate(algorithms):
        mse_values = [stats[shape][algo]['mse'] for shape in shapes]
        offset = (i - n_algos/2) * width
        ax.bar(x + offset, mse_values, width, label=algo)
    
    ax.set_xlabel('Shapes', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('MSE Comparison Across Algorithms', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_metric_heatmap(stats_path: str, output_path: str):
    stats = load_summary_stats(stats_path)
    
    if not stats:
        print("No statistics found!")
        return
    
    shapes = list(stats.keys())
    algorithms = list(stats[shapes[0]].keys()) if shapes else []
    
    chamfer_matrix = np.zeros((len(shapes), len(algorithms)))
    mse_matrix = np.zeros((len(shapes), len(algorithms)))
    
    for i, shape in enumerate(shapes):
        for j, algo in enumerate(algorithms):
            chamfer_matrix[i, j] = stats[shape][algo]['chamfer']
            mse_matrix[i, j] = stats[shape][algo]['mse']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    im1 = axes[0].imshow(chamfer_matrix, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(np.arange(len(algorithms)))
    axes[0].set_yticks(np.arange(len(shapes)))
    axes[0].set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(shapes, fontsize=10)
    axes[0].set_title('Chamfer Distance Heatmap', fontsize=14, fontweight='bold')
    
    for i in range(len(shapes)):
        for j in range(len(algorithms)):
            text = axes[0].text(j, i, f'{chamfer_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(mse_matrix, cmap='YlOrRd', aspect='auto')
    axes[1].set_xticks(np.arange(len(algorithms)))
    axes[1].set_yticks(np.arange(len(shapes)))
    axes[1].set_xticklabels(algorithms, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels(shapes, fontsize=10)
    axes[1].set_title('MSE Heatmap', fontsize=14, fontweight='bold')
    
    for i in range(len(shapes)):
        for j in range(len(algorithms)):
            text = axes[1].text(j, i, f'{mse_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def generate_comparison_visualizations(base_path: str = '/home/jaswinjose/Desktop/PCD'):
    stats_path = os.path.join(base_path, 'stats', 'summary_report.csv')
    
    comparison_path = os.path.join(base_path, 'stats', 'algorithm_comparison.png')
    plot_algorithm_comparison(stats_path, comparison_path)
    
    heatmap_path = os.path.join(base_path, 'stats', 'metric_heatmap.png')
    plot_metric_heatmap(stats_path, heatmap_path)
    


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = '/home/jaswinjose/Desktop/PCD'
    generate_summary_report(base_path)
    generate_comparison_visualizations(base_path)
    