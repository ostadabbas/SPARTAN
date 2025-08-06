import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
import os
import json

import sys
sys.path.append("..")

from sharegpt4v import share4v_val_dataset
from model import longclip
import numpy as np

ckpt_pth = "./checkpoints/etfl/"

testset = share4v_val_dataset('../config/val_v2.txt')
testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=1, pin_memory=True)

model, _ = longclip.load_from_clip("ViT-B/16", device='cpu',download_root=None, use_fft=True, new_loss=True)
# model_pth = list(os.listdir("./checkpoints_abspose/"))
model_pth = list(os.listdir(ckpt_pth))
if len(model_pth) == 0:
    print("No checkpoint found")

# ep_ckpt = [int(model.split("-")[0]) for model in model_pth]
ep_ckpt = [model for model in model_pth]
ep_ckpt.sort()
model_pth_full = ckpt_pth + ep_ckpt[-1]
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.logit_scale = torch.nn.Parameter(torch.ones([]) * 4.6052)  
model = model.cuda()
model.load_state_dict(torch.load(model_pth_full))

result = []

def array_to_string(array):
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if array.shape != (24, 8):
        raise ValueError("Input array must have shape (24, 8)")
    return '\n'.join([' '.join(map(str, row)) for row in array])

@torch.no_grad()
def test_epoch(dataloader):
    all_correct = []
    all_correct_5 = []
    all_total = []

    j_all_correct = []
    j_all_correct_5 = []
    j_all_total = []

    i = 0
    for images, text, _, poses, filename in tqdm(dataloader):
        correct = 0
        correct_5 = 0
        total = 0
        j_correct = 0
        j_correct_5 = 0
        j_total = 0
        image = images.cuda()
        image = image.permute(0,2,1,3,4)
        b, t, c, h, w = image.size()
        # Remove the batch dimensions
        image = image.reshape(-1, c, h, w)
        image_features = model.encode_image(image)
        image_features = image_features.view(b, t, -1)  # [B, T, 512]
        # Now take the mean along the temporal direction
        image_features = image_features.mean(dim=1, keepdim=False)  # image features are now ready
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_token = longclip.tokenize(text, truncate=True).cuda()
        text_feature = model.encode_text(text_token, poses.cuda().float())
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        batch_dict = {}
        batch_dict["files"] = filename
        batch_dict["retrieival"] = []
        for i in range(text_feature.shape[0]):
            single_dict = {}
            single_dict["text"] = text[i] 
            single_dict["pose"] = array_to_string(poses[i].cpu().numpy())
            sim = text_feature[i] @ image_features.T
            sim = sim.squeeze()
            # result.append("sim "+str(i)+":"+str(sim))
            correct_i = torch.argmax(sim)
            top_5_indices = torch.topk(sim, 5).indices
            single_dict["GT"] = str(i)
            single_dict["Pred"] = str(correct_i)
            batch_dict["retrieival"].append(single_dict)
            if i==correct_i:
                correct = correct + 1
            if i in top_5_indices:
                correct_5 += 1
            total = total + 1
        
        result.append(batch_dict)
        print("t2i Batch {}: acc@1: {}, acc@5: {}".format(i, correct/total, correct_5/total))
        all_correct.append(correct/total)
        all_correct_5.append(correct_5/total)
        all_total.append(total)

        for j in range(image_features.shape[0]):
            sim = image_features[j] @ text_feature.T
            sim = sim.squeeze()
            
            correct_j = torch.argmax(sim)
            top_5_indices = torch.topk(sim, 5).indices
            
            if j == correct_j:
                j_correct += 1
            if j in top_5_indices:
                j_correct_5 += 1
            j_total += 1
        print("i2t Batch {}: acc@1: {}, acc@5: {}".format(i, j_correct/j_total, j_correct_5/j_total))
        j_all_correct.append(j_correct/j_total)
        j_all_correct_5.append(j_correct_5/j_total)

        i += 1

    return np.array(all_correct), np.array(all_correct_5), np.array(j_all_correct), np.array(j_all_correct_5)

co, co5, jco, jco5 = test_epoch(testloader)

print("t2i final mean acc@1: {}, std acc@1: {}, mean acc@5: {}, std acc@5:{}".format(np.mean(co), np.std(co), np.mean(co5), np.std(co5)))
print("i2t final mean acc@1: {}, std acc@1: {}, mean acc@5: {}, std acc@5:{}".format(np.mean(jco), np.std(jco), np.mean(jco5), np.std(jco5)))

with open("test_result_fallback_1.json", "w") as f:
    json.dump(result, f)

with open("test_result_{}.json".format(ckpt_pth.split('/')[-2]), "w") as f:
    json.dump(result, f)