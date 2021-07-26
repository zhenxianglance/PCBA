from __future__ import print_function
import numpy as np
import torch.utils.data
from sklearn.neighbors import NearestNeighbors


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def knn_dist(points, N, k=2):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    dist, _ = nbrs.kneighbors(points)
    dist = dist.sum(axis=1)/k

    dist_mean = dist.sum()/N
    dist_median = np.median(dist)

    return dist, dist_mean, dist_median


def create_points_RS(center, points, npoints):
    Total_trial = 30
    k = 4
    _, _, median_knn_target = knn_dist(points, points.shape[0], k)

    optima = float('inf')

    for radius in np.arange(start=0.01, stop=0.5, step=0.01):   # [0.01:0.01:0.5]
        for trial in range(Total_trial):
            points_trial = np.zeros([npoints, 3])
            for n in range(npoints):
                theta = np.random.uniform(0, np.pi)
                phi = np.random.uniform(0, 2 * np.pi)
                points_trial[n, 0] = radius * np.sin(theta) * np.cos(phi)
                points_trial[n, 1] = radius * np.sin(theta) * np.sin(phi)
                points_trial[n, 2] = radius * np.cos(theta)

            dist, _, median = knn_dist(points_trial, npoints, k)
            MAD = np.median(np.abs(dist-median_knn_target))
            if MAD < optima:
                optima = MAD
                points_adv = points_trial
                optimal_r = radius

    return points_adv + center
