# Adapt from https://github.com/PaulAlbert31/PLS/blob/main/utils.py#L52
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as Data
import os
import pdb
from tqdm import tqdm
import argparse

# Image.open is very slow, resize and convert them into numpy first

def preprocess_miniimagenet(noise_rate, base_path = '/home/mjtang/dataset/cnwl/mini-imagenet'):
    size = 32
    target_savepoint = f"data/cnwl/red_{noise_rate}_images.npy"
    target_label_savepoint = f"data/cnwl/red_{noise_rate}_labels.npy"
    split_file = os.path.join(base_path, f"split/red_noise_nl_{noise_rate}")        
    if os.path.exists(target_savepoint):
        print(f"image numpy checkpoint exist: {target_savepoint}")
        return
    os.makedirs("data/cnwl", exist_ok=True)
    Resize = transforms.Resize(size, interpolation=Image.BICUBIC)
    images, image_path, image_label = [], [], []
    with open(split_file,"r") as f:
        for line in f:
            x, y = line.strip().split()
            image_path.append(x)
            image_label.append(int(y))
    
    np.save(target_label_savepoint, image_label)
    
    for image in tqdm(image_path, 'load image'):    
        with Image.open(os.path.join(base_path, f'validation/{image}')) as img:
            images.append(Resize(img).convert('RGB'))
    
    images_np = np.array([np.array(image) for image in images])
    np.save(target_savepoint, images_np)
    
    # mean = [0.4728, 0.4487, 0.4031]
    # std = [0.2744, 0.2663 , 0.2806]
    # size = 32
    
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(size, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std),
    # ])
        
    # test_transform = transforms.Compose([
    #     transforms.CenterCrop(size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    # pdb.set_trace()
def preprocess_valid_miniimagenet(base_path = '/home/mjtang/dataset/cnwl/mini-imagenet'):
    size = 32
    target_savepoint = f"data/cnwl/red_valid_images.npy"
    target_label_savepoint = f"data/cnwl/red_valid_labels.npy"
    split_file = os.path.join(base_path, "split/clean_validation")        
    if os.path.exists(target_savepoint):
        print(f"image numpy checkpoint exist: {target_savepoint}")
        return
    os.makedirs("data/cnwl", exist_ok=True)
    Resize = transforms.Resize(size, interpolation=Image.BICUBIC)
    images, image_path, image_label = [], [], []
    with open(split_file,"r") as f:
        for line in f:
            x, y = line.strip().split()
            image_path.append(x)
            image_label.append(int(y))
    
    np.save(target_label_savepoint, image_label)
    
    for image, label in tqdm(zip(image_path, image_label), 'load image'):    
        with Image.open(os.path.join(base_path, f'validation/{label}/{image}')) as img:
            images.append(Resize(img.convert('RGB')))
    
    images_np = np.array([np.array(image) for image in images])
    np.save(target_savepoint, images_np)


def load_miniimagenet(noise_rate, base_path = '/home/mjtang/dataset/cnwl/mini-imagenet'):
    size = 32

    target_savepoint = f"data/cnwl/red_{noise_rate}_images.npy"
    target_label_savepoint = f"data/cnwl/red_{noise_rate}_labels.npy"
    
    image_label = np.load(target_label_savepoint)
    images = np.load(target_savepoint)
    images = [Image.fromarray(image) for image in images]
    
    mean = [0.4728, 0.4487, 0.4031]
    std = [0.2744, 0.2663 , 0.2806]
    size = 32
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
        
    test_transform = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # pdb.set_trace()
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

def cosine_lr(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs

    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))

    return lrs

import torch
from torch import optim
def test_lr():
    lr1 = cosine_lr(0.1, 1, 200)
    plt.plot(lr1)
    plt.savefig('image1.jpg')

    model = torch.nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 定义scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=200)
    lr2 = []
    # 开始训练循环
    for epoch in range(200):
        current_lr = optimizer.param_groups[0]['lr']
        lr2.append(current_lr)

        optimizer.zero_grad()
        output = model(torch.randn(1, 2))
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # 调整学习率
        scheduler.step()
        
    plt.plot(lr2)
    plt.savefig('image2.jpg')
    print(np.all(np.array(lr1)==np.array(lr2)))
    print(lr1[:10])
    print(lr2[:10])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_rate', type=float, default=0.6) # 0.0 for val set
    parser.add_argument('--valid', action='store_true')
    args, others = parser.parse_known_args()
    if args.valid:
        preprocess_valid_miniimagenet()
    else:
        preprocess_miniimagenet(args.noise_rate)
    # load_miniimagenet(args.noise_rate)
    # pass