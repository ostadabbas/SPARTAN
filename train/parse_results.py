import json
import numpy as np
from PIL import Image
import os
import threading
from evo.core import metrics
from evo.core.units import Unit
from evo.tools import file_interface
from evo.core import sync
import evo.core.trajectory as trajectory

def parse_pose_str(pose_str, output_file):
    rows = pose_str.strip().split("\n")
    matrix = np.array([list(map(float, row.split())) for row in rows])
    # Replace the first column with a range of float numbers starting at 0.0 with an increment of 0.1
    matrix[:, 0] = np.arange(0.0, 0.1 * matrix.shape[0], 0.1)[:matrix.shape[0]]
    # Save the matrix to a text file with 6 decimal places
    # output_file = "output_matrix.txt"
    np.savetxt(output_file, matrix, fmt="%.6f", delimiter=" ")

home_dir = "/work/zura-storage/Data/DrivingSceneDDM/images/"
result_dir = "./vis_results_etfl/"

json_file = 'test_result_etfl.json'
with open(json_file, 'r') as f:
    data = json.load(f)

def process_batch(bidx, batch):
    for idx, files in enumerate(batch["files"]):
        batch_dir = result_dir + "batch_{:03d}".format(bidx) + "/videos/"
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        gif_path = batch_dir + "/{:06d}.gif".format(idx)
        if os.path.exists(gif_path):
            continue
        dataset, filename = files.split("/")
        start_idx = int(filename.split(".")[0])
        frames = []
        for i in range(start_idx, start_idx + 24):
            complete_pth = home_dir + dataset + "/{:06d}.png".format(i)
            frames.append(np.array(Image.open(complete_pth).resize((256, 256))))

        # Save frames as a GIF
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:], duration=24, loop=0)

    total_ct = 0
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_mean = 0.0
    gt_traj = 0.0
    
    for ridx, payload in enumerate(batch["retrieival"]):
        traj_pth = result_dir + "batch_{:03d}".format(bidx) + "/traj/retrieval_{:03d}/".format(ridx)
        if not os.path.exists(traj_pth):
            os.makedirs(traj_pth)
        with open(traj_pth + "desc.txt", "w") as f:
            f.write(payload["text"])
        if payload["GT"] == payload["Pred"].split(",")[0].split("(")[-1]:
            with open(traj_pth + "eq.txt", "w") as f:
                parse_pose_str(payload["pose"], traj_pth + "eq.txt")
            traj_ref = file_interface.read_tum_trajectory_file(traj_pth + "eq.txt")
            gt_traj += traj_ref.path_length
        else:
            pred_it = int(payload["Pred"].split(",")[0].split("(")[-1])
            with open(traj_pth + "gt.txt", "w") as f:
                parse_pose_str(payload["pose"], traj_pth + "gt.txt")
            with open(traj_pth + "pred.txt", "w") as f:
                parse_pose_str(batch["retrieival"][pred_it]["pose"], traj_pth + "pred.txt")
            ref_file = traj_pth + "gt.txt"
            est_file = traj_pth + "pred.txt"
            traj_ref = file_interface.read_tum_trajectory_file(ref_file)
            traj_est = file_interface.read_tum_trajectory_file(est_file)
            gt_traj += traj_ref.path_length
            max_diff = 0.01
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
            ape_metric.process_data((traj_ref, traj_est))
            ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
            ape_mean += ape_stat
            
        total_ct += 1
    
    print("Batch {} processed with APE {}m, relative {}%".format(bidx, ape_mean/total_ct,ape_mean*100/gt_traj))

threads = []
for bidx, batch in enumerate(data):
    thread = threading.Thread(target=process_batch, args=(bidx, batch))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
                