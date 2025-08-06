import os
import json
import random
from tqdm import tqdm
import random

annotation_loc = "/work/zura-storage/Data/DrivingSceneDDM/ann_frame"
video_loc = "/work/zura-storage/Data/DrivingSceneDDM/images"
total_frame = 24
entry_list = []
val_ratio = 0.05
slid_window_max = 12


# we first find videos
for dataset in tqdm(os.listdir(video_loc)):
    if dataset[0] != "c":
        continue
    dataset_path = os.path.join(video_loc, dataset)
    if int(dataset.split("_")[-1]) <= 8:
        annotation_file = os.path.join(annotation_loc, dataset, f"parsed_answers.txt")
        with open(annotation_file, "r") as annotation_f:
            annotations = annotation_f.readlines()
        max_len = len(annotations)-24
        i = 0
        while i < max_len:
            frame_str, desc = annotations[i].split(" {'<MORE_DETAILED_CAPTION>': ")
            frame_path = os.path.join(dataset_path, frame_str)
            assert os.path.exists(frame_path), frame_path
            ann_text = desc[1:-2].replace("image", "video").replace("\n", " ").replace('"', '').replace("'", '')
            frame_path = "/".join(frame_path.split("/")[-2:])
            write_str = f"{frame_path} *** {ann_text}\n"
            entry_list.append(write_str)
            i += random.randint(1, slid_window_max)
    else:
        annotation_file = os.path.join(annotation_loc, dataset+".json") #use json
        with open(annotation_file, "r") as annotation_f:
            annotations = json.load(annotation_f)
        i = 0
        frame_list = list(os.listdir(dataset_path))
        max_len = len(frame_list)-24
        while i < max_len:
            frame = frame_list[i]
            img_full_key = dataset_path + "/" + frame
            frame_path = os.path.join(dataset_path, frame)
            if img_full_key not in annotations.keys():
                continue
            ann_text = annotations[img_full_key].replace("image", "video").replace("\n", " ").replace('"', '').replace("'", '')
            frame_path = "/".join(frame_path.split("/")[-2:])
            write_str = f"{frame_path} *** {ann_text}\n"
            entry_list.append(write_str)
            i += random.randint(1, slid_window_max)

random.shuffle(entry_list)

fw = open("./train_v2.txt", "w")
fv = open("./val_v2.txt", "w")
for write_str in entry_list:
    if random.random() < val_ratio:
        fv.write(write_str)
    else:
        fw.write(write_str)

fw.close()
fv.close()