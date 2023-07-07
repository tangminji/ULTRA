import numpy as np
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch

def dataset_split_without_noise(train_images, train_labels, split_per=0.9, seed=1):
    total_labels = train_labels[:, np.newaxis]

    num_samples = int(total_labels.shape[0])
    np.random.seed(seed)
    train_set_index = np.random.choice(num_samples, int(num_samples * split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)
    print(train_images.shape)
    train_set, val_set = train_images[train_set_index], train_images[val_set_index]
    train_labels, val_labels = total_labels[train_set_index], total_labels[val_set_index]

    return train_set, val_set, train_labels.squeeze(), val_labels.squeeze()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean)
    print(std)
    return mean, std


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, device, seed):
    # index -> noise index
    # n -> noise_rate
    # dataset -> mnist, cifar10, cifar100 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1

    label_num = num_classes
    np.random.seed(seed)
    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    # flip_distribution = stats.beta(a=0.01, b=(0.01 / n) - 0.01, loc=0, scale=1)
    flip_rate = flip_distribution.rvs(labels.shape[0])#generate random variables

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)

    W = np.random.randn(label_num, feature_size, label_num)

    labels = labels.to(device)
    W = torch.FloatTensor(W).to(device)

    for i, (x, y) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.to(device)
        A = x.view(1, -1).mm(W[y].squeeze(0)).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    # np.save("transition_matrix.npy", P)

    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    print(f'noise rate = {(new_label != np.array(labels.cpu())).mean()}')

    return np.array(new_label)