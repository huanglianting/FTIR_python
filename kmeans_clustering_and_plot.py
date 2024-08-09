import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_clustering_and_plot(spectrum_1, spectrum_2, x, save_path, n_clusters=7, max_k=20):
    # Combine the spectra for clustering
    combined_spectrum = np.hstack((spectrum_1, spectrum_2))

    # Perform K-means clustering for different values of k to find the optimal number of clusters
    cost = []
    models = []
    K = range(1, max_k)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(combined_spectrum.T)
        cost.append(kmeans.inertia_)
        models.append(kmeans)

    # Plot the cost function to use the elbow method
    plt.figure()
    plt.plot(K, cost, marker='o')
    plt.xticks(ticks=np.arange(1, max_k, 2), labels=np.arange(1, max_k, 2))  # 设置 x 轴刻度
    plt.xlabel('Number of Clusters')
    plt.ylabel('Cost Function')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'KMeans_Elbow_Method.png'))
    plt.show()

    # Perform K-means clustering with the specified number of clusters (default is 7)
    kmeans = models[n_clusters - 1]
    clusters = kmeans.predict(combined_spectrum.T)

    # Colors for each cluster
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # Plot the mean absorption spectra for each cluster
    plt.figure()
    for i in range(n_clusters):
        cluster_data = combined_spectrum[:, clusters == i]
        mean_spectrum = np.mean(cluster_data, axis=1)
        plt.plot(x, mean_spectrum, color=colors[i % len(colors)], label=f'Cluster {i + 1}')
    plt.gca().invert_xaxis()
    plt.xlabel('Wave-number (cm-1)')
    plt.ylabel('Absorbance')
    plt.title('Mean Absorption Spectra for Each Cluster')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'Mean_Absorption_Spectra.png'))
    plt.show()
