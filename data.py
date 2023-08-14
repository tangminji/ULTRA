import os

import numpy as np
import torch.utils.data as Data
import PIL
from PIL import Image
import tools
import torch
import data_process
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import transformer
from utils import TwoCropTransform, ThreeCropTransform

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
        if noise_rate!='valid':
            self.train_real_labels = np.load('{}/data/cnwl/red_{}_real.npy'.format(path, noise_rate))
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

# Old
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

def get_miniimagenet_dataset(args):
    mean = [0.4728, 0.4487, 0.4031]
    std = [0.2744, 0.2663 , 0.2806]
    size = 32
    
    train_wtransform = transforms.Compose([
        # transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_stransform = transforms.Compose([
        # transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.RandomResizedCrop(size, interpolation=PIL.Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        # transforms.Resize(size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if args.model_type == 'ours_cl':
        train_dataset = MiniImagenet_dataset(args.path, noise_rate=args.noise_rate2,
                                             transform=ThreeCropTransform(train_wtransform, train_stransform),
                                             target_transform=transformer.transform_target,)
    else:
        train_dataset = MiniImagenet_dataset(args.path, noise_rate=args.noise_rate2,
                                              transform=train_wtransform,
                                              target_transform=transformer.transform_target,)

    test_dataset = MiniImagenet_dataset(args.path, noise_rate='valid',
        transform=test_transform,
        target_transform=transformer.transform_target)

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

Clothing1M_PATH="dataset/Clothing_1M"

class Clothing1M(VisionDataset):
    def __init__(self, root, mode='train', transform=None, target_transform=None, num_per_class=-1):

        super(Clothing1M, self).__init__(root, transform=transform, target_transform=target_transform)

        if mode == 'train':
            flist = os.path.join(root, "annotations/noisy_train.txt")#100w,265664(18976*14)
        if mode == 'val':
            flist = os.path.join(root, "annotations/clean_val.txt")#14313
        if mode == 'test':
            flist = os.path.join(root, "annotations/clean_test.txt")#10526

        self.impaths, self.targets = self.flist_reader(flist)

        rng = np.random.RandomState(seed=0)
        if num_per_class > 0:
            impaths, targets = [], []
            num_each_class = np.zeros(14)
            indexs = np.arange(len(self.impaths))
            rng.shuffle(indexs)

            for i in indexs:
                if num_each_class[self.targets[i]] < num_per_class:
                    impaths.append(self.impaths[i])
                    targets.append(self.targets[i])
                    num_each_class[self.targets[i]] += 1

            self.impaths, self.targets = impaths, targets
            print('#samples/class: {};\n#total samples: {:d}\n'.format([int(i) for i in num_each_class],
                                                                       int(sum(num_each_class))))

        # TODO
        self.targets = np.array(self.targets)
        self.train_labels = self.targets
    #         # for quickly ebug
    #         self.impaths, self.targets = self.impaths[:1000], self.targets[:1000]

    def __getitem__(self, index):
        impath = self.impaths[index]
        target = self.train_labels[index]

        img = Image.open(impath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.impaths)

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


class Clothing1MWithIdx(Clothing1M):
    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 num_per_class=-1):
        super(Clothing1MWithIdx, self).__init__(root=root,
                                                mode=mode,
                                                transform=transform,
                                                target_transform=target_transform,
                                                num_per_class=num_per_class)
    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (sample, target, index) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return img, target, index

def get_Clothing1M_train_and_val_loader(args):
    '''
    batch_size: train(32),val/test(128)
    '''
    print('==> Preparing data for Clothing1M..')

    train_transform = transforms.Compose([transforms.Resize((256)),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                                          ])
    test_transform = transforms.Compose([transforms.Resize((256)),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
                                         ])
    train_dataset = Clothing1MWithIdx(root=Clothing1M_PATH,
                                      mode='train',
                                      transform=train_transform,
                                      target_transform=transformer.transform_target,
                                      num_per_class=args.num_per_class)

    val_dataset = Clothing1MWithIdx(root=Clothing1M_PATH,
                                    mode='val',
                                    transform=test_transform,
                                    target_transform=transformer.transform_target,)
    test_dataset = Clothing1MWithIdx(root=Clothing1M_PATH,
                                     mode='test',
                                     transform=test_transform,
                                     target_transform=transformer.transform_target,)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4 * args.batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4 * args.batch_size, shuffle=False)

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
    