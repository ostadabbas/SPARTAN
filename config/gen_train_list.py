import os
import json
import random

annotation_loc = "/work/zura-storage/Data/DrivingSceneDDM/annotations"
pose_loc = "/work/zura-storage/Data/DrivingSceneDDM/poses"
video_loc = "/work/zura-storage/Data/DrivingSceneDDM/videos_pose"
total_frame = 24
val_list = []
val_ratio = 0.05

with open("./train.txt", "w") as f:
    # we first find videos
    for dataset in os.listdir(video_loc):
        dataset_path = os.path.join(video_loc, dataset)
        annotation_file = os.path.join(annotation_loc, dataset, f"grok_output_summaries.json")
        annotations = None
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as annotation_f:
                annotations = json.load(annotation_f)
        else:
            new_ann_file = os.path.join(annotation_loc, dataset+".json")
            with open(new_ann_file, "r") as annotation_f:
                annotations2 = json.load(annotation_f)
        for video in os.listdir(dataset_path):
            video_path = os.path.join(dataset_path, video)
            video_path = "/".join(video_path.split("/")[-2:])
            # based on file name calculate json index
            start_idx = str(int(video.split(".")[0]) * total_frame)
            end_idx = str(int(video.split(".")[0]) * total_frame + total_frame - 1)
            ann = start_idx + "-" + end_idx
            if annotations is None:
                img_idx = int(video.split(".")[0]) * total_frame
                img_full_key = "/work/zura-storage/Data/DrivingSceneDDM/images/" + dataset + "/" + "{:06d}.png".format(img_idx)
                ann_text = annotations2[img_full_key].replace("image", "video")
                annotation = ann_text
            else:
                annotation = annotations[ann] # get annotation for this video
            annotation = annotation.replace("\n", " ")
            # write to file
            write_str = f"{video_path} *** {annotation} *** {start_idx},{end_idx}\n"
            if random.random() < val_ratio:
                val_list.append(write_str)
            f.write(write_str)

with open("./val.txt", "w") as f:
    for line in val_list:
        f.write(line)