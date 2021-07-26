from __future__ import print_function
import argparse
import os
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--attack_dir', type=str, default='attack', help='attack folder')

opt = parser.parse_args()

# Visualize training data
attack_data_train = np.load(os.path.join(opt.attack_dir, 'attack_data_train.npy'))
backdoor_pattern_train = np.load(os.path.join(opt.attack_dir, 'backdoor_pattern_train.npy'))
for i in range(len(attack_data_train)):
    points_clean = attack_data_train[i]
    points_backdoor = backdoor_pattern_train[i]
    points_clean = points_clean[:len(points_clean)-len(points_backdoor), :]
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points_clean)
    pcd1.paint_uniform_color(np.array([0.1, 0.1, 0.8]))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points_backdoor)
    pcd2.paint_uniform_color(np.array([0.8, 0.1, 0.1]))
    # Attack
    o3d.visualization.draw_geometries([pcd1, pcd2])
    # Clean
    o3d.visualization.draw_geometries([pcd1])




