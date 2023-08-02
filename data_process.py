import numpy as np
import torch
import tools

def open_closed_noisy_labels(dataset1, dataset1_label, dataset2, device, closed_noise_type='symmetric', openset_noise_rate=0.2, closed_set_noise_rate=0.2, num_classes=10, random_seed=1):
    # dataset1 is corrupted by dataset2 
    # dataset1 and dataset2 .npy format
    # not -> dataset1 and dataset2 do not have same classes, e.g., CIFAR-10 and SVHN (MNIST, *CIFAR-100)

    num_total_1, num_total_2 = int(dataset1.shape[0]), int(dataset2.shape[0])
    real_label = np.copy(dataset1_label)

    noise_rate = float(openset_noise_rate + closed_set_noise_rate)
    num_noisy_labels_1 = int(noise_rate * num_total_1)
    num_open_noisy_labels_1, num_closed_noisy_labels_1 = int(openset_noise_rate * num_total_1), int(closed_set_noise_rate * num_total_1)

    np.random.seed(random_seed)
    corrupted_labels_index_1, corrupted_labels_index_2 = np.random.choice(num_total_1, num_noisy_labels_1, replace=False), np.random.choice(num_total_2, num_open_noisy_labels_1, replace=False)
    corrupted_open_noisy_labels_index_1, corrupted_closed_noisy_labels_index_1  = corrupted_labels_index_1[:num_open_noisy_labels_1], corrupted_labels_index_1[num_open_noisy_labels_1:]
    print(corrupted_open_noisy_labels_index_1)
    print(corrupted_closed_noisy_labels_index_1)

    # open_set_corruption (images corruption)
    dataset1[corrupted_open_noisy_labels_index_1] = dataset2[corrupted_labels_index_2]
    real_label[corrupted_open_noisy_labels_index_1] = num_classes  # OOD

    # closed_set_corruption (labels corruption)
    labels = dataset1_label[corrupted_closed_noisy_labels_index_1]
    labels = labels[:, np.newaxis]
    if closed_noise_type == 'instance':
        feature_size = 3*32*32
        norm_std=0.1
        dataset = torch.from_numpy(dataset1[corrupted_closed_noisy_labels_index_1]).float()
        labels = torch.from_numpy(labels)
        data_labels = zip(dataset, labels)
        noisy_labels = tools.get_instance_noisy_label(1.0, data_labels, labels, num_classes, feature_size, norm_std, device, random_seed)
        dataset1_label[corrupted_closed_noisy_labels_index_1] = noisy_labels.squeeze()
    return dataset1, dataset1_label, real_label




