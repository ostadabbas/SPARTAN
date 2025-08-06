import os

with open(os.path.join(os.path.dirname(__file__), 'train.txt'), 'r') as file:
    train_text = file.readlines()

for i in range(len(train_text)):
    train_text[i] = train_text[i].strip()
    train_text[i] = train_text[i].split(' *** ')
    poses = train_text[i][-1]
    pose_start, pose_end = poses.split(',')
    if int(pose_end) - int(pose_start) != 23:
        print(train_text[i])