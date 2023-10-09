import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import DataLoader, Dataset
from data_partition import partition_data
import pandas as pd
from sklearn.model_selection import train_test_split

class Cifar_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(Cifar_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class Ppda_dataset(data.Dataset):
    def __init__(self, data, labels):
        super(Ppda_dataset, self).__init__()
        print((data))
        data = data.astype(np.float32)
        labels = labels.astype(np.float32)
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)

def cifar_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    if dataset == "cifar10":
        train_dataset = CIFAR10(base_path, True, download=True)
        test_dataset = CIFAR10(base_path, False, download=True)
        transform_train=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == "cifar100":
        train_dataset = CIFAR100(base_path, True, download=True)
        test_dataset = CIFAR100(base_path, False, download=True)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            normalize])
        
    train_image = train_dataset.data
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    val_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i][:int(0.8*len(net_dataidx_map[i]))]
        val_idxs = net_dataidx_map[i][int(0.8*len(net_dataidx_map[i])):]
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = Cifar_Truncated(data=train_image[val_idxs], labels=train_label[val_idxs], transform=transform_test)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions

def ppda_dataset_read(dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    if dataset == "iris":
        df = pd.read_csv("Documents/btp/pFedGraph/data/iris/bezdekIris.data", sep = ",", header = None)
        print(df.head())
        X = df.to_numpy()
        print(X.shape)
        X = X[:,:4]
        Y = np.ones((150,))
        Y[:50] = 0
        Y[100:150] = 2
        print(X.shape)
        print(Y.shape)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    train_image = X_train
    train_label = Y_train
    test_image = X_test
    test_label = Y_test
    print(type(train_image[0][0]))
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    n_train = train_label.shape[0]
    net_dataidx_map, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    val_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i][:int(0.8*len(net_dataidx_map[i]))]
        val_idxs = net_dataidx_map[i][int(0.8*len(net_dataidx_map[i])):]
        train_dataset = Ppda_dataset(data=train_image[train_idxs], labels=train_label[train_idxs])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = Ppda_dataset(data=train_image[val_idxs], labels=train_label[val_idxs])
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_dataloaders.append(train_loader)
        val_dataloaders.append(val_loader)
    
    test_dataset = Ppda_dataset(data=test_image, labels=test_label)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, val_dataloaders, test_loader, net_dataidx_map, traindata_cls_counts, data_distributions
    
