import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import random
import decord
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import ToPILImage
import copy
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import sys
import random


data4v_root = '/work/zura-storage/Data/DrivingSceneDDM/'
json_name = 'annotations/'
image_root = data4v_root + 'images/'
train_list = '../config/train_v2.txt'
val_list = '../config/val_v2.txt'
# train_list = '../config/train_short.txt'
# val_list = '../config/val_short.txt'
pose_location = "/work/zura-storage/Data/DrivingSceneDDM/poses/"

def convert_image_to_rgb(image):
    return image.convert("RGB")


def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def transform_video(video_tensor, n_px):
    # video_tensor = video_tensor.permute(0, 3, 1, 2)
    transform_fn = transform(n_px)
    to_pil_image = ToPILImage()
    transformed_frames = [transform_fn(to_pil_image(video_tensor[t, :, :, :])) for t in range(video_tensor.shape[0])]
    transformed_video_tensor = torch.stack(transformed_frames, dim=0)
    return transformed_video_tensor.permute(1, 0, 2, 3)


class longclip_shared_dataset(data.Dataset):
    def __init__(self):
        self.pose_lib = {}
        for pose_file in os.listdir(pose_location):
            # look for the npy file to open
            pose_np = np.load(osp.join(pose_location, pose_file))
            self.pose_lib[pose_file] = pose_np
        model, _ = clip.load("ViT-B/16")
        self.n_px = model.visual.input_resolution

    def cvt_npy_str_list(self, npy_file):
        npy_file_mod = npy_file - npy_file[0,:]
        ret_arr = []
        for row in npy_file_mod:
            ret_arr.append(' '.join([f"{float(x):.3f}" for x in row]))
        return ret_arr
    
    def cvt_str_list_str(self, str_list):
        return '\n'.join(str_list)
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        line = self.json_data[index]
        line_split = line.strip().split(" *** ")
        # print(line_split)
        filename, prompt = line_split[0], line_split[1]
        pose_start = int(filename.split("/")[1].split(".")[0])
        short_prompt = prompt.split(".")[0]
        dataset_name = filename.split("/")[0] + "_poses.npy"
        pose_end = pose_start + 23
        copy_pose = None
        try:
            curr_pose_npy = self.pose_lib[dataset_name][int(pose_start):int(pose_end)+1]
            copy_pose = copy.deepcopy(curr_pose_npy)
            copy_pose[:,:4] -= curr_pose_npy[0,:4]
            copy_pose = copy_pose.astype(np.float16)
            if np.any(np.isnan(copy_pose)) or np.any(np.isinf(copy_pose)):
                print("Pose contains NaN or INF values", filename, dataset_name, pose_start, pose_end)
                sys.exit(1)
        except:
            print("Error in reading pose file", filename, dataset_name, pose_start, pose_end)
            sys.exit(1)
        if len(curr_pose_npy) < 24:
            print(line)
        # print(curr_pose_npy.shape)
        # curr_pose_list = self.cvt_npy_str_list(curr_pose_npy)
        # pose_ret = self.cvt_str_list_str(curr_pose_list)

        frames = []
        for i in range(24):
            frame_path = osp.join(image_root, filename.split("/")[0], "{:06d}.png".format(pose_start+i))
            frame = Image.open(frame_path)
            frames.append(frame)
        
        video_tensor = torch.stack([ToTensor()(frame) for frame in frames])
        # print(video_tensor.shape)
        # video_tensor = video_tensor.permute(1, 0, 2, 3)  # Convert to [C, T, H, W]
        video_tensor = transform_video(video_tensor, self.n_px)
        # print(video_tensor.shape, curr_pose_npy.shape)

        return video_tensor, prompt, short_prompt, copy_pose, filename

class share4v_val_dataset(longclip_shared_dataset):
    def __init__(self, val_list_param):
        super().__init__()
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        with open(val_list_param, 'r') as fp:
            self.json_data = fp.readlines()
        self.total_len = len(self.json_data)  




class share4v_train_dataset(longclip_shared_dataset):
    def __init__(self):
        super().__init__()
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        with open(train_list, 'r') as fp:
            self.json_data = fp.readlines()
        self.total_len = len(self.json_data)
