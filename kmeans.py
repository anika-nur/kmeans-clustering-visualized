'''NAME: NUR-E-JANNAT ANIKA
'''


import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.data = None
        self.centroids = None
        self.labels = None
        self.num_samps = None
        self.num_features = None
        self.inertia_value = float('inf') 

    def set_data(self, data):
        self.data = np.array(data)
        self.num_samps = self.data.shape[0]       
        self.num_features = self.data.shape[1]
    
    def get_data(self):
        return np.copy(self.data)
    
    def dist_pt_to_pt(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def dist_pt_to_centroids(self, pt, centroids):
        return np.sqrt(np.sum((centroids - pt) ** 2, axis=1))

    def initialize(self, k):
        self.k = k
        indices = np.random.choice(self.data.shape[0], self.k, replace=False)
        self.centroids = self.data[indices]
        self.inertia_value = float('inf')
        return self.centroids
    
    def update_labels(self, centroids):
        self.centroids = centroids
        distances = np.array([self.dist_pt_to_centroids(pt, self.centroids) for pt in self.data])
        self.labels = np.argmin(distances, axis=1)
        return self.labels
    
    def update_centroids(self, k, labels, old_centroids):
        new_centroids = np.zeros_like(old_centroids)

        for i in range(k):
            cluster_points = self.data[labels == i]

            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = old_centroids[i]

        diff_from_prev = new_centroids - old_centroids

        self.centroids = new_centroids
        return new_centroids, diff_from_prev


    def inertia(self):
        total_distance = 0
        for i, pt in enumerate(self.data):
            centroid = self.centroids[self.labels[i]]
            total_distance += np.sum((pt - centroid) ** 2)
        return total_distance / len(self.data)
    
    def cluster(self):
        self.initialize(self.k)  

        for _ in range(self.max_iters):
            old_centroids = np.copy(self.centroids)

            self.labels = self.update_labels(self.centroids)

            self.centroids, _ = self.update_centroids(self.k, self.labels, old_centroids)

            if np.allclose(old_centroids, self.centroids):
                break

        self.data_centroid_labels = self.labels
        return self.labels


    def plot_clusters(self):
        colors = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E']

        plt.figure(figsize=(6, 5))

        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        distances = np.linalg.norm(grid_points[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        Z = np.argmin(distances, axis=1)
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')

        for i in range(self.k):
            cluster_points = self.data[self.labels == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                s=40,
                color=colors[i % len(colors)],
                edgecolor='k',
                label=f"Cluster {i + 1}"
            )

        plt.scatter(
            self.centroids[:, 0],
            self.centroids[:, 1],
            c='black',
            marker='x',
            s=120,
            label='Centroids'
        )

        for i, (x, y) in enumerate(self.centroids):
            plt.text(
                x + 0.05, y + 0.05,
                f"C{i + 1}",
                fontsize=10,
                color='black',
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
            )

        plt.title("K-Means Clustering Results with Decision Boundaries")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            frameon=False
        )
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()


    def compute_inertia(self):
        total_distance = 0.0
        for i, pt in enumerate(self.data):
            cluster_idx = self.data_centroid_labels[i]
            centroid = self.centroids[cluster_idx]
            total_distance += np.sum((pt - centroid) ** 2)
        mean_sq_distance = total_distance / len(self.data)
        return mean_sq_distance
    
    def elbow_plot(self, max_k = 10, n_iter = 1):
        inertias = []
        k_values = range(1, max_k + 1)

        for k in k_values:
            _, _, best_inertia = self.cluster_batch(k = k, n_iter = n_iter)
            inertias.append(best_inertia)

        plt.figure(figsize=(6, 5))
        plt.plot(k_values, inertias, 'o-', linewidth=2)
        plt.title(f"Elbow Plot with {n_iter} runs per k)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.xticks(k_values)
        plt.grid(True, linestyle = '--', alpha = 0.5)

    def cluster_batch(self, k, n_iter=10):
        best_inertia = float('inf')
        best_labels = None
        best_centroids = None

        for i in range(n_iter):
            np.random.seed(i)
            self.k = k
            self.initialize(self.k)
            labels = self.update_labels(self.centroids)
            centroids, _ = self.update_centroids(k, labels, self.centroids)

            self.centroids = centroids
            self.data_centroid_labels = labels
            inertia_val = self.compute_inertia()

            if inertia_val < best_inertia:
                best_inertia = inertia_val
                best_labels = labels
                best_centroids = centroids

        self.labels = best_labels
        self.centroids = best_centroids
        self.data_centroid_labels = best_labels

        return best_labels, best_centroids, best_inertia
    
    def replace_color_with_centroid(img_flat, labels, centroids):
        """
        Replace every pixel's RGB value with its cluster centroid's RGB value.

        Parameters
        ----------
        img_flat : ndarray, shape=(num_pixels, 3)
            Flattened image array.
        labels : ndarray, shape=(num_pixels,)
            Cluster label for each pixel.
        centroids : ndarray, shape=(k, 3)
            Centroid RGB values from K-Means.

        Returnss
        -------
        quantized_flat : ndarray, shape=(num_pixels, 3)
            Quantized (color-replaced) flattened image.
        """
        quantized_flat = centroids[labels]      
        return quantized_flat

