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
        with Image.open(os.path.join(base_path, f'all_images/{image}')) as img:
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

def check_miniimagenet(noise_rate, base_path = '/home/mjtang/dataset/cnwl/mini-imagenet'):
    size = 32
    target_label_savepoint = f"data/cnwl/red_{noise_rate}_labels.npy"
    real_label_savepoint = f"data/cnwl/red_{noise_rate}_real.npy"
    
    if noise_rate != 'valid':
        split_file = os.path.join(base_path, f"split/red_noise_nl_{noise_rate}")
        std_map = {}
        labels, real_labels = [], []
        for mode in ['red', 'blue']:
            std_file = os.path.join(base_path, f"split/{mode}_noise_nl_0.0")
            with open(std_file,"r") as f:
                for line in f:
                    x, y = line.strip().split()
                    std_map[x] = int(y)
        with open(split_file,"r") as f:
            for line in f:
                x, y = line.strip().split()
                y = int(y)
                labels.append(y)
                real_y = std_map.get(x, 100)
                real_labels.append(real_y) # 100代表OOD
        old_labels = np.load(target_label_savepoint)
        acc = np.sum(np.array(real_labels) == np.array(labels))
        assert np.all(labels == old_labels), f"labels: {labels}\n old: {old_labels}"
        print(f'{noise_rate} Acc: {acc}/{len(real_labels)}={acc/len(real_labels)}')
        np.save(real_label_savepoint, real_labels)

    target_savepoint = f"data/cnwl/red_{noise_rate}_images.npy"
    target_old_savepoint = f"data/cnwl-old/red_{noise_rate}_images.npy"
    img1 = np.load(target_savepoint)
    img2 = np.load(target_old_savepoint)
    print('same as old', (img1.shape==img2.shape) & np.all(img1 == img2))

def get_class_name(base_path = '/home/mjtang/dataset/cnwl/mini-imagenet'):
    with open(os.path.join(base_path, "class_name.txt"), "r") as f:
        class_name = f.read().strip().splitlines()
    print(len(class_name))
    print(class_name)
    return class_name

Clothing1M_PATH="/home/ttwu/script/dataset/Clothing_1M"

def preprocess_clothing1m(mode='train', num_per_class=-1, root=Clothing1M_PATH):
    print(f'process clothing1m {mode} set')
    if mode == 'train':
        flist = os.path.join(root, "annotations/noisy_train.txt")#100w,265664(18976*14)
    if mode == 'val':
        flist = os.path.join(root, "annotations/clean_val.txt")#14313
    if mode == 'test':
        flist = os.path.join(root, "annotations/clean_test.txt")#10526
    target_savepoint = f"data/clothing1m/{mode}_{num_per_class}_images.npy" if mode != 'train' else f"data/clothing1m/{mode}_{num_per_class}/"
    target_label_savepoint = f"data/clothing1m/{mode}_{num_per_class}_labels.npy"

    if os.path.exists(target_savepoint):
        print(f"image numpy checkpoint exist: {target_savepoint}")
        return
    os.makedirs("data/clothing1m", exist_ok=True)
    impaths = []
    targets = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            row = line.split(" ")
            impaths.append(root + '/' + row[0][7:])#remove "image/"
            targets.append(int(row[1]))
    rng = np.random.RandomState(seed=0)
    if num_per_class > 0:
        impaths2, targets2 = [], []
        num_each_class = np.zeros(14)
        indexs = np.arange(len(impaths))
        rng.shuffle(indexs)

        for i in indexs:
            if num_each_class[targets[i]] < num_per_class:
                impaths2.append(impaths[i])
                targets2.append(targets[i])
                num_each_class[targets[i]] += 1

        impaths, targets = impaths2, targets2
        print('#samples/class: {};\n#total samples: {:d}\n'.format([int(i) for i in num_each_class],
                                                                    int(sum(num_each_class))))
    np.save(target_label_savepoint, targets)
    print('label saved')
    images = []
    Resize = transforms.Resize((256))

    if mode != 'train':
        for image in tqdm(impaths, 'load image'):
            with Image.open(image) as img:
                images.append(Resize(img.convert('RGB')))
        images_np = np.array([np.array(image) for image in images])
        print('data shape:', images_np.shape)
        np.save(target_savepoint, images_np)
        print('images saved')
    else:
        os.makedirs(target_savepoint, exist_ok=True)
        for index, image in tqdm(enumerate(impaths), 'load image'):
            with Image.open(image) as img:
                image = np.array(Resize(img.convert('RGB')))
                np.save(f"{target_savepoint}/{index}.npy", image)
        print('image saved')                

from torchvision.datasets import VisionDataset
class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, num_per_class=-1):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)
        target_savepoint = f"data/clothing1m/{mode}_{num_per_class}_images.npy" if mode != 'train' else f"data/clothing1m/{mode}_{num_per_class}/"
        target_label_savepoint = f"data/clothing1m/{mode}_{num_per_class}_labels.npy"
        self.train_labels = np.load(target_label_savepoint)
        self.mode = mode
        if mode != 'train':
            self.train_data = np.load(target_savepoint, allow_pickle=True)
            self.train_data = [Image.fromarray(x) for x in self.train_data]
        else:
            self.train_data = []
            self.image_folder = target_savepoint

    def __getitem__(self, index):
        target = self.train_labels[index]
        if self.mode != 'train':
            img = self.train_data[index]
        else:
            img = Image.fromarray(np.load(f"{self.image_folder}/{index}.npy"))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_labels)

    def flist_reader(self, flist):
        impaths = []
        targets = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impaths.append(self.root + '/' + row[0][7:])#remove "image/"
                targets.append(int(row[1]))
        return impaths, targets

    def update_corrupted_label(self, noise_label):
        self.train_labels[:] = noise_label[:]

def test_clothing1m():
    train_transform = transforms.Compose([ #transforms.Resize((256)),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                                          ])
    test_transform = transforms.Compose([ #transforms.Resize((256)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                                         ])
    dataset = Clothing1M(Clothing1M_PATH, 'train', num_per_class=18976 )
    print('===train 5===')
    img, label = dataset[5]
    print(label)
    print(img.size)
    print(img)
    img.save('train5.jpg')
    print('===train 6===')
    img, label = dataset[6]
    print(label)
    print(img.size)
    print(img)
    img.save('train6.jpg')

    dataset = Clothing1M(Clothing1M_PATH, 'test')
    
    print('===test 5===')
    img, label = dataset[5]
    print(label)
    print(img.size)
    print(img)
    img.save('test5.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_rate', type=float, default=0.6) # 0.0 for val set
    parser.add_argument('--valid', action='store_true')
    args, others = parser.parse_known_args()
    

    # preprocess_clothing1m('train', num_per_class=18976)
    # preprocess_clothing1m('val')
    # preprocess_clothing1m('test')

    # get_class_name()
    check_miniimagenet(0.2)
    check_miniimagenet(0.4)
    check_miniimagenet(0.6)
    check_miniimagenet(0.8)
    check_miniimagenet('valid')
    # if args.valid:
    #     preprocess_valid_miniimagenet()
    # else:
    #     preprocess_miniimagenet(args.noise_rate)
    # load_miniimagenet(args.noise_rate)
    # pass    