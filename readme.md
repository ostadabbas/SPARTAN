# SPARTAN: Spatiotemporal Pose-Aware Retrieval for Text-guided Autonomous Navigation

Implementation of our camera pose-text joint video retreival model. The paper is accepted at BMVC 2025. This repository is based on the existing work Long-CLIP[⇱](https://github.com/beichenzbc/Long-CLIP) on GitHub.

## Dataset
We present DrivingScenePTX, a comprehensive driving video dataset that includes both frame-wise camera poses and rich textual scene descriptions.

### Download
- [RGB Images (~131GB)](https://zzzura-secure.duckdns.org/downloads/poseclip_images.zip)
- [Annotations](https://zzzura-secure.duckdns.org/downloads/poseclip_ann.zip)
- [Camera Pose](https://zzzura-secure.duckdns.org/downloads/poseclip_pose.zip)

You will need to manually place them into the correct folder:
```shell
- dataset
|-images
 |-carla_01
 |-...
|-poses
 |-carla_01.npy
 |-...
|-annotations
 |-carla_01
  |-grok_output_summaries.json
 |-...
 |-carla_09.json
 |-...
```
and then edit the corresponding variables in ```train/sharegpt4v.py```, ```train/train.py```.

## Setup

This repository has been tested on Ubuntu 22.04 with CUDA 12.1. First, make sure CUDA is properly installed and configured. Install ```torch==2.3.0``` based on your machine and GPU configurations[⇱](https://pytorch.org/get-started/previous-versions/). Then, execute the following commands.

```shell
pip install -r requirements.txt
```

## Backbone Preperation

For training from scratch, download and put LongCLIP-B [⇱](https://huggingface.co/BeichenZhang/LongCLIP-B) and LongCLIP-L [⇱](https://huggingface.co/BeichenZhang/LongCLIP-L) models based on CLIP VIT-B/16 and CLIP VIT-L/14 in ```checkpoint``` folder. For inference, download our pretrained model [⇱](https://zzzura-secure.duckdns.org/downloads/9-0.972-longclip.zip) and place it under ```train/checkpoints``` folder.

## Model Training and Fine-tuning

- To start training from scratch, run:

```shell
torchrun train.py --batch-size 14 --epochs 400 --fft --new_loss
```

- Additionally, if you wish to start with an existing (downloaded) checkpoint, run
```shell
torchrun train.py --batch-size 14 --epochs 400 --fft --new_loss --resume
```

## Model Evaluation

All of the videos for evaluation should be listed in ```config/val_v2.txt```. Please first modify ```ckpt_pth``` and ```testset``` varible in ```train/test.py``` and run ``` python test.py ```. The script will create ```test_result_{epoch}.json``` file under your current directory. 

To visualize the retrieval, edit ```parse_results.py``` by changing ```home_dir``` to where the validation images are located, and change ```results_dir``` to a place you would like to save the videos. Finally, edit ```json_file``` varible to point to the json file just created and run ```python parse_results.py```.

To enable novel SLAM-based evaluation, edit ```parse_results_slam.py``` by pointing ```vis_folder``` varible to the ```results_dir``` in the last step and run ```python parse_results_slam.py```.


## Citation

```bibtex
@inproceedings{bai2025clip,
    author = {Bai, Xiangyu and Anish Sreeramagiri, Sai and Siddhartha Vivek Dhir Rangoju, Sai and Galoaa, Bishoy and Ostadabbas, Sarah},
    title = {SPARTAN: Spatiotemporal Pose-Aware Retrieval for Text-guided Autonomous Navigation},
    booktitle = {British Machine Vision Conference (BMVC)},
    year = {2025}
}
```
