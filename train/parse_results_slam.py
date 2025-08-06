# use this if you have vis_results folder ready
import os
from evo.core import metrics
from evo.core.units import Unit
from evo.tools import file_interface
from evo.core import sync
import evo.core.trajectory as trajectory
import numpy as np

vis_folder = "/work/zura-storage/Workspace/poseclip/Long-CLIP/train/vis_results_etfl"

total_ct = 0
ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
pose_relation = metrics.PoseRelation.rotation_angle_deg

# normal mode
delta = 1
delta_unit = Unit.frames

# all pairs mode
all_pairs = False 
rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
ape_mean = 0.0
gt_traj = 0.0
apes = []
rpes = []
gts = []

batch_list = list(os.listdir(vis_folder))
for item in batch_list:
    traj_pth = os.path.join(vis_folder, item, "traj")
    for ret in os.listdir(traj_pth):
        if os.path.exists(os.path.join(traj_pth, ret, "eq.txt")):
            traj_ref = file_interface.read_tum_trajectory_file(os.path.join(traj_pth, ret, "eq.txt"))
            if traj_ref.path_length - 0.0 > 1e-3:
                gts.append(traj_ref.path_length)
                apes.append(0.0)
                rpes.append(0.0)
        else:
            assert os.path.exists(os.path.join(traj_pth, ret, "pred.txt"))
            traj_ref = file_interface.read_tum_trajectory_file(os.path.join(traj_pth, ret, "gt.txt"))
            traj_est = file_interface.read_tum_trajectory_file(os.path.join(traj_pth, ret, "pred.txt"))
            max_diff = 0.01
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
            if traj_ref.path_length - 0.0 > 1e-3:
                gts.append(traj_ref.path_length)
                ape_metric.process_data((traj_ref, traj_est))
                rpe_metric.process_data((traj_ref, traj_est))
                ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
                rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
                apes.append(ape_stat)
                rpes.append(rpe_stat)
                ape_mean += ape_stat
        gt_traj += traj_ref.path_length
        total_ct += 1

gts = np.array(gts)
apes = np.array(apes)
rpes = np.array(rpes)

print("ape: mean(m): {}, std(m): {}, mean(%): {}, std(%): {}".format(ape_mean/total_ct, np.std(apes), np.mean(apes/gts), np.std(apes/gts)))
print("rpe: mean(m): {}, std(m): {}, mean(%): {}, std(%): {}".format(np.mean(rpes), np.std(rpes), np.mean(rpes/gts), np.std(rpes/gts)))

