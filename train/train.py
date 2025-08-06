import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime, timedelta
from torch.cuda.amp import GradScaler
# import warnings
# warnings.filterwarnings("ignore")


class CLIP_Clean_Train():
    def __init__(self, rank,local_rank,args):
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root, use_fft=args.fft, new_loss=args.new_loss)
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.exp_name = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()
        self.max_acc = 0.0
        self.curr_acc = 0.0


    def train_epoch(self, dataloader, epoch, start_time, start_iter=0):
        print("Epoch: ", epoch)
        running_loss = 0.0
        running_loss_short = 0.0
        #rank = torch.distributed.get_rank() 
        num_batches_per_epoch = len(dataloader)
        for i, (images, texts, short_text, poses, filename) in enumerate(pbar:=tqdm(dataloader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue
            texts = longclip.tokenize(texts, truncate=True).cuda()
            short_text = longclip.tokenize(short_text, truncate=True).cuda()
            self.scheduler(step)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_long,loss_short = self.model(images, texts,short_text,poses,self.rank)
                loss=loss_long+loss_short
            running_loss += loss.item()
            running_loss_short += loss_short.item()
            pbar.set_description(f"loss: {running_loss / (i + 1):.4f}, short_loss: {running_loss_short / (i + 1):.4f}, epoch {epoch}")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if datetime.now() - start_time > timedelta(hours=23.8):
                if self.rank == 0:
                    name = "-{:.3f}-longclip.pt".format(self.curr_acc)
                    torch.save(self.model.module.state_dict(), './checkpoints/'+str(epoch)+name)

        

    @torch.no_grad()
    def test_epoch(self, dataloader):
        rank = torch.distributed.get_rank()
        correct = 0
        total = 0

        for images, text, _, poses, filename in tqdm(dataloader, disable=(rank != 0)):

            image = images.cuda()
            image = image.permute(0,2,1,3,4)
            b, t, c, h, w = image.size()
            # Remove the batch dimensions
            image = image.reshape(-1, c, h, w)
            image_features = self.model.module.encode_image(image)
            image_features = image_features.view(b, t, -1)  # [B, T, 512]
            # Now take the mean along the temporal direction
            image_features = image_features.mean(dim=1, keepdim=False)  # image features are now ready

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text = longclip.tokenize(text, truncate=True).cuda()
            text_feature = self.model.module.encode_text(text, poses.cuda().float())
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            for i in range(text_feature.shape[0]):
                text = text_feature[i]
                sim = text @ image_features.T
                sim = sim.squeeze()
                # print(sim)
                correct_i = torch.argmax(sim)

                if i==correct_i:
                    correct = correct + 1
                total = total + 1

        return correct/total
    
    def test(self, epoch=0):
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.model.eval()
            testset = share4v_val_dataset('../config/val_v2.txt')
            testloader = torch.utils.data.DataLoader(testset, batch_size=12, num_workers=1, pin_memory=True)
            with torch.no_grad():    

                acc = self.test_epoch(testloader)
                print("=====================================")
                print(f"test mean of share4v retrieval: {acc}")
                print("=====================================")

            return acc
    
    def train(self, resume=False, warmup_length=200):
        trainset = share4v_train_dataset()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=1, pin_memory=True)
        resume_iter = 0
        resume_epoch = 0
        start_time = datetime.now()
        if resume:
            model_pth = list(os.listdir("./checkpoints/"))
            if len(model_pth) == 0:
                print("No checkpoint found")
                return
            # ep_ckpt = [int(model.split("long")[0]) for model in model_pth]
            ep_ckpt = []
            ep_dict = {}
            use_new_format = False
            acc_str = "0.000"
            for model in model_pth:
                appx = model.split("longclip")[0]
                if appx[-1] == "-":
                    ep_ckpt.append(int(appx.split("-")[0]))
                    use_new_format = True
                    acc_str = appx.split("-")[1]
                    ep_dict[ep_ckpt[-1]] = acc_str
                else:
                    ep_ckpt.append(int(appx))

            ep_ckpt.sort()
            if use_new_format:
                model_pth_full = "./checkpoints/" + str(ep_ckpt[-1]) + "-" + acc_str + "-longclip.pt"
                self.max_acc = float(ep_dict[ep_ckpt[-1]])
                self.curr_acc = self.max_acc
            else:
                model_pth_full = "./checkpoints/" + str(ep_ckpt[-1]) + "longclip.pt"
            assert os.path.exists(model_pth_full), "Checkpoint not found"
            resume_epoch = ep_ckpt[-1] + 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # model, _ = longclip.load(model_pth, device=device)
            self.model.module.load_state_dict(torch.load(model_pth_full))
             
        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = resume_epoch
        print("resume from epoch: ", start_epoch, ", with current acc: ", self.curr_acc)
        
        for epoch in range(start_epoch, self.num_epoch):
            self.train_epoch(train_loader, epoch, start_time, start_iter=resume_iter)
            self.curr_acc = self.test()
            # if datetime.now() - start_time > timedelta(hours=23.5) or epoch == self.num_epoch - 1 or self.curr_acc > self.max_acc:
            if self.rank == 0:
                name = "-{:.3f}-longclip.pt".format(self.curr_acc)
                if self.curr_acc > self.max_acc:
                    self.max_acc = self.curr_acc
                os.makedirs('./checkpoints/'+self.exp_name, exist_ok=True)
                torch.save(self.model.module.state_dict(), './checkpoints/'+self.exp_name+'/'+str(epoch)+name)
            # if self.rank == 0 and epoch % 10 == 0:
            #     name = "longclip.pt"
            #     #torch.distributed.barrier()
            #     torch.save(self.model.module.state_dict(), './checkpoints/'+str(epoch)+name)

def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ and False:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank, rank % num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )
    parser.add_argument(
        "--fft",
        default=False,
        action='store_true',
        help="include fast fourier transform."
    )
    parser.add_argument(
        "--new_loss",
        default=False,
        action='store_true',
        help="include the new loss function."
    )
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    args = parser.parse_args()
    rank,local_rank = setup_distributed()
    print("DDP Done")

    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)
