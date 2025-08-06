import numpy as np

nppath = "/work/zura-storage/Data/DrivingSceneDDM/poses/carla_10_poses.npy"

poses = np.load(nppath)
poses[:,:4] = poses[:,:4] - poses[0,:4]
output_path = "./carla_10_poses.txt"

with open(output_path, "w") as f:
    for pose in poses:
        f.write(" ".join(f"{value:.6f}" for value in pose) + "\n")