import os

import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
import data_process
import torchvision.transforms as transforms
import transformer
from utils import TwoCropTransform

class cifar10_svhn_dataset(Data.Dataset):
    def __init__(self, path, device, seed, train=True, transform=None, target_transform=None, noise_type='symmetric', noise_rate1=0.2,
                 noise_rate2=0.2, split_per=0.9):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        dataset1, labels1 = np.load('{}/data/cifar10/train_images.npy'.format(path)), np.load('{}/data/cifar10/train_labels.npy'.format(path))
        dataset2, labels2 = np.load('{}/data/svhn/train_images.npy'.format(path)), np.load('{}/data/svhn/train_labels.npy'.format(path))

        noisy_dataset1, noisy_labels1, real_labels = data_process.open_closed_noisy_labels(dataset1, labels1, dataset2, device,
                                                                              closed_noise_type=noise_type,
                                                                              openset_noise_rate=noise_rate1,
                                                                              closed_set_noise_rate=noise_rate2,
                                                                              num_classes=10, random_seed=seed)

        self.train_data, self.val_data, self.train_labels, self.val_labels, self.train_real_labels, self.val_real_labels = tools.dataset_split_without_noise(
            noisy_dataset1, noisy_labels1, real_labels, split_per, seed)

        save_path = f'{path}/data_meta/cifar10s/{noise_rate1}_{noise_rate2}/{seed}'
        os.makedirs(save_path, exist_ok=True)
        np.save(f"{save_path}/train_labels.npy",self.train_labels)
        np.save(f"{save_path}/val_labels.npy",self.val_labels)
        np.save(f"{save_path}/train_real_labels.npy",self.train_real_labels)
        np.save(f"{save_path}/val_real_labels.npy",self.val_real_labels)
        

        if self.train:      
            self.train_data = self.train_data.reshape((45000,3,32,32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        
        else:
            self.val_data = self.val_data.reshape((5000,3,32,32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))
        
    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)

    def update_corrupted_label(self, noise_label):
        self.train_labels[:] = noise_label[:]

class MiniImagenet_dataset(Data.Dataset):
    def __init__(self, path, noise_rate='valid', transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.train_data = np.load('{}/data/cnwl/red_{}_images.npy'.format(path, noise_rate))
        self.train_labels = np.load('{}/data/cnwl/red_{}_labels.npy'.format(path, noise_rate))
        self.train_data = [Image.fromarray(image) for image in self.train_data]
    
    def __getitem__(self, index):
        
        img, label = self.train_data[index], self.train_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.train_data)
    
    def update_corrupted_label(self, noise_label):
        self.train_labels[:] = noise_label[:]
# load dataset
#def load_data(args):

# Adapt from https://github.com/PaulAlbert31/PLS/blob/main/utils.py#L52
def get_miniimagenet_dataset(args):
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
    if args.model_type == 'ours_cl':
        train_dataset = MiniImagenet_dataset(args.path, noise_rate=args.noise_rate2,
                                             transform=TwoCropTransform(train_transform),
                                             target_transform=transformer.transform_target,
                                             )
    else:
        train_dataset = MiniImagenet_dataset(args.path, noise_rate=args.noise_rate2,
                                              transform=train_transform,
                                              target_transform=transformer.transform_target,
                                              )
    test_dataset = MiniImagenet_dataset(args.path, noise_rate='valid',
        transform=test_transform,
        target_transform=transformer.transform_target
    )
    val_dataset = test_dataset # follow PLS
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               drop_last=False,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             drop_last=False,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              shuffle=False)
    return train_loader, val_loader, test_loader

def get_cifars_dataset(args):

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    val_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if args.model_type == 'ours_cl':
        train_dataset = cifar10_svhn_dataset(args.path, args.device, args.seed, True,
                                             transform=TwoCropTransform(train_transform),
                                             target_transform=transformer.transform_target,
                                             noise_type=args.noise_type,
                                             noise_rate1=args.noise_rate1,
                                             noise_rate2=args.noise_rate2,
                                             split_per=args.split_per)
    else:
        train_dataset = cifar10_svhn_dataset(args.path, args.device, args.seed, True,
                                              transform=train_transform,
                                              target_transform=transformer.transform_target,
                                              noise_type=args.noise_type,
                                              noise_rate1=args.noise_rate1,
                                              noise_rate2=args.noise_rate2,
                                              split_per=args.split_per)

    val_dataset = cifar10_svhn_dataset(args.path, args.device, args.seed, False,
                                            transform=val_transform,
                                            target_transform=transformer.transform_target,
                                            noise_type=args.noise_type,
                                            noise_rate1=args.noise_rate1,
                                            noise_rate2=args.noise_rate2,
                                            split_per=args.split_per)

    test_dataset = cifar10_test_dataset(args.path,
        transform=test_transform,
        target_transform=transformer.transform_target
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               drop_last=False,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             drop_last=False,
                                             shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              drop_last=False,
                                              shuffle=False)

    return train_loader, val_loader, test_loader






class cifar10_test_dataset(Data.Dataset):
    def __init__(self, path, transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform
           
        self.test_data = np.load('{}/data/cifar10/test_images.npy'.format(path))
        self.test_labels = np.load('{}/data/cifar10/test_labels.npy'.format(path))
        self.test_data = self.test_data.reshape((10000,3,32,32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
        print(self.test_data.shape)
    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label, index
    
    def __len__(self):
        return len(self.test_data)


class clothing1m_dataset(Data.Dataset):
    def __init__(self, train=True, transform=None, target_transform=None, split_per=0.9, random_seed=1, num_class=14):
        self.imgs = np.load('images/index/noisy_train_images.npy')
        self.labels = np.load('images/index/noisy_train_labels.npy')
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split_clothing(self.imgs,
                                                                                                          self.labels,
                                                                                                          split_per,
                                                                                                          random_seed,
                                                                                                          num_class)

        
        self.train_labels = self.train_labels.squeeze()
        self.val_labels = self.val_labels.squeeze()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        if self.train:
            fn = self.train_data[index]
            img = Image.open(fn).convert('RGB')
            label = self.train_labels[index]
        else:
            fn = self.val_data[index]
            img = Image.open(fn).convert('RGB')
            label = self.val_labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.val_labels)

class clothing1m_train_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = np.load('images/index/noisy_train_images.npy')
        self.labels = np.load('images/index/noisy_train_labels.npy')
        self.train_labels = self.labels
        self.train_data = self.imgs

        
        #self.train_labels = self.train_labels.squeeze()
        #self.transform = transform
        #self.target_transform = target_transform

        #self.train_data = []
        #self.train_labels = []
        #num_samples = 50000
        #class_num = torch.zeros(14)
        #for i in range(self.imgs.shape[0]):
            #label = self.labels[i]
            #if class_num[label] < (num_samples/14) and len(self.train_data) < num_samples:
                #self.train_data.append(self.imgs[i])
                #self.train_labels.append(self.labels[i])
                #class_num[label] += 1


    def __getitem__(self, index):
        
        fn = self.train_data[index]
        img = Image.open(fn).convert('RGB')
        label = self.train_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index 

    def __len__(self):
        return len(self.train_labels)

class clothing1m_val_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.imgs = np.load('images/index/clean_val_images.npy')
        self.labels = np.load('images/index/clean_val_labels.npy')
        self.train_labels = self.labels
        self.train_data = self.imgs

        self.train_labels = self.train_labels.squeeze()
        self.transform = transform
        self.target_transform = target_transform
        

    def __getitem__(self, index):
        
        fn = self.train_data[index]
        img = Image.open(fn).convert('RGB')
        label = self.train_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index 

    def __len__(self):
        return len(self.train_labels)

class clothing1m_test_dataset(Data.Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.imgs = np.load('images/index/clean_test_images.npy')
        self.labels = np.load('images/index/clean_test_labels.npy')
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):
        return len(self.imgs)
    