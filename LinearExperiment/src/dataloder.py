import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_svmlight_file
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from argument import argument
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
import os

class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset).__init__()
        self.data = data
        self.label = label
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.data.shape[0]

def random_Subsequence(lenth, total_lenth):
    delete_sequence = random.sample(range(0, total_lenth), lenth)
    delete_sequence.sort()
    remain_seqeunce = [i for i in range(total_lenth) if i not in delete_sequence]
    new_sequence = np.concatenate([delete_sequence, remain_seqeunce])
    if len(new_sequence) != total_lenth:
        raise Exception('random Subsequence error !')
    return new_sequence

def Load_binaryMnist(delete_num=None, shuffle=False):
    """
    return two tuple [(train_data, train_label), (test_data, test_label)]
    """
    dataroot = '../data'
    train_data = datasets.MNIST(dataroot, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))
    test_data = datasets.MNIST(dataroot, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))

    # choose train 1/7 samples
    subset_indices = ((train_data.targets == 1) + (train_data.targets == 7)).nonzero().view(-1)
    data_train = train_data.data[subset_indices]
    data_train = data_train.clone().detach() / 255.0
    label_train = train_data.targets[subset_indices]
    print(f"train labels: {label_train}")
    for index, item in enumerate(label_train):
        if item == 1:
            label_train[index] = -1
        else:
            label_train[index] = 1
    label_train.unsqueeze_(dim=1)
    # random shuffling the train data for random delete
    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, data_train.shape[0])
        data_train = data_train[re_sequence]
        label_train = label_train[re_sequence]
        
    # choose test 1/7 samples
    subset_indices = ((test_data.targets == 1) + (test_data.targets == 7)).nonzero().view(-1)
    data_test = test_data.data[subset_indices]
    data_test = data_test.clone().detach() / 255.0
    label_test = test_data.targets[subset_indices]
    for index, item in enumerate(label_test):
        if item == 1:
            label_test[index] = -1
        else:
            label_test[index] = 1
    label_test.unsqueeze_(dim=1)

    return (data_train, label_train), (data_test, label_test), re_sequence

def Load_covtype(delete_num, shuffle=False, isL2_norm=True):
    """
    pre-process should be done for faster loading data
    """
    train_file_path = '../data/COVTYPE/train_data.pt'
    test_file_path = '../data/COVTYPE/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making dataset...')
        X, y = load_svmlight_file('../data/COVTYPE/covtype.libsvm.binary.scale.bz2')
        # chage label from (1, 2) to (-1, 1)
        y = (y - 1.5) * 2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
        X_train = torch.from_numpy(X_train.todense()).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test.todense()).float()
        y_test = torch.from_numpy(y_test).long()

        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
        print('Done !')
    else:
        print('Loading ...')
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)
    
    # if isL2_norm:
    #     X_train /= X_train.norm(p=2, dim=1).max()
    #     X_test /= X_test.norm(p=2, dim=1).max()

    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]
    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence



def Load_epsilon(delete_num, shuffle=False, isL2_norm=True):
    train_file_path = '../data/EPSILON/train_data.pt'
    test_file_path = '../data/EPSILON/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making dataset...')
        X_train, y_train = load_svmlight_file('../data/EPSILON/epsilon_normalized.bz2')
        X_test, y_test = load_svmlight_file('../data/EPSILON/epsilon_normalized.t.bz2')

        min_max_scaler = MinMaxScaler()

        X_train = torch.from_numpy(min_max_scaler.fit_transform(X_train.todense())).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(min_max_scaler.fit_transform(X_test.todense())).float()
        y_test = torch.from_numpy(y_test).long()

        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
        print('Done !')

    else:
        print('Loading...')
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)
    # if isL2_norm:
    #     print('loading ...')
    #     X_train /= X_train.norm(p=2, dim=1).max()
    #     X_test /= X_test.norm(p=2, dim=1).max()

    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]
    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence


def Load_gisette(delete_num, shuffle=False, isL2_norm=True):
    train_file_path = '../data/GISETTE/train_data.pt'
    test_file_path = '../data/GISETTE/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making data set ...')
        X_train, y_train = load_svmlight_file('../data/GISETTE/gisette.bz2')
        X_test, y_test = load_svmlight_file('../data/GISETTE/gisette_test.bz2')

        X_train = torch.from_numpy(X_train.todense()).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test.todense()).float()
        y_test = torch.from_numpy(y_test).long()

        X_train = (X_train + 1) / 2
        X_test = (X_test + 1) / 2

        print('done !')
        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
    else:
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)
    
    # if isL2_norm:
    #     print('loading ...')
    #     X_train /= X_train.norm(p=2, dim=1).max()
    #     X_test /= X_test.norm(p=2, dim=1).max()

    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]
    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence

def Load_ijcnn1(shuffle=False, isL2_norm=False):
    train_file_path = '../data/IJCNN1/train_data.pt'
    test_file_path = '../data/IJCNN1/test-data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making data set ...')

        X_train, y_train = load_svmlight_file('../data/IJCNN1/ijcnn1.bz2')
        X_test, y_test = load_svmlight_file('../data/IJCNN1/ijcnn1_test.bz2')

        min_max_scaler = MinMaxScaler()

        X_train = torch.from_numpy(min_max_scaler.fit_transform(X_train.todense())).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(min_max_scaler.fit_transform(X_test.todense())).float()
        y_test = torch.from_numpy(y_test).long()

        print('done !')
        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
    else:
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)

    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1))

def Load_higgs(delete_num, shuffle=False, isL2_norm=True):
    train_file_path = '../data/HIGGS/train_data.pt'
    test_file_path = '../data/HIGGS/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making data set ...')
        X, y = load_svmlight_file('../data/HIGGS/higgs.bz2')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

        y_train = (y_train - 0.5) / 0.5
        y_test = (y_test - 0.5) / 0.5

        min_max_scaler = MinMaxScaler()

        X_train = torch.from_numpy(min_max_scaler.fit_transform(X_train.todense())).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(min_max_scaler.fit_transform(X_test.todense())).float()
        y_test = torch.from_numpy(y_test).long()

        print('done !')
        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
    else:
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)

    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]
    
    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence
    
def Load_madelon(delete_num, shuffle):
    train_file_path = '../data/MADELON/train_data.pt'
    test_file_path = '../data/MADELON/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making data set ...')
        X_train, y_train = load_svmlight_file('../data/MADELON/train.txt')
        X_test, y_test = load_svmlight_file('../data/MADELON/test.txt')
        
        min_max_scaler = MinMaxScaler()

        X_train = torch.from_numpy(min_max_scaler.fit_transform(X_train.todense())).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(min_max_scaler.fit_transform(X_test.todense())).float()
        y_test = torch.from_numpy(y_test).long()

        print('done !')
        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
    else:
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)

    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]

    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence

def Load_phishing(delete_num, shuffle=False):
    train_file_path = '../data/PHISHING/train_data.pt'
    test_file_path = '../data/PHISHING/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making data set ...')
        X, y= load_svmlight_file('../data/PHISHING/data.txt')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)

        y_train = (y_train - 0.5) / 0.5
        y_test = (y_test - 0.5) / 0.5

        X_train = torch.from_numpy(X_train.todense()).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test.todense()).float()
        y_test = torch.from_numpy(y_test).long()

        print('done !')
        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
    else:
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)

    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]

    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence


def Load_splice(delete_num, shuffle=False):
    train_file_path = '../data/SPLICE/train_data.pt'
    test_file_path = '../data/SPLICE/test_data.pt'

    if os.path.exists(train_file_path) == False or os.path.exists(test_file_path) == False:
        print('Making data set ...')
        X_train, y_train = load_svmlight_file('../data/SPLICE/train.txt')
        X_test, y_test = load_svmlight_file('../data/SPLICE/test.txt')
        
        min_max_scaler = MinMaxScaler()

        X_train = torch.from_numpy(min_max_scaler.fit_transform(X_train.todense())).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(min_max_scaler.fit_transform(X_test.todense())).float()
        y_test = torch.from_numpy(y_test).long()

        print('done !')
        torch.save([X_train, y_train], f=train_file_path)
        torch.save([X_test, y_test], f=test_file_path)
    else:
        X_train, y_train = torch.load(train_file_path)
        X_test, y_test = torch.load(test_file_path)
    
    re_sequence = None
    if shuffle == True:
        re_sequence = random_Subsequence(delete_num, X_train.shape[0])
        X_train = X_train[re_sequence]
        y_train = y_train[re_sequence]

    return (X_train, y_train.unsqueeze(dim=1)), (X_test, y_test.unsqueeze(dim=1)), re_sequence



def make_loader(data, batch_size, head=-1, rear=-1, shuffle=False):
    """
    data is a tuple(image, label)
    head : the data we want to start, if -1, we set head = 0
    rear : the data we want to end, if -1 , we set rear = the real rear
    """
    len = data[0].shape[0]
    if head == -1:
        head = 0
    if rear == -1:
        rear = len

    image = data[0][head : rear]
    label = data[1][head : rear]

    sets = MyDataset(image, label)

    return DataLoader(sets, batch_size=batch_size, shuffle=shuffle)

def Load_Data(args, delete_num=None, shuffle=False):

    if args.dataset == 'binaryMnist':
        return Load_binaryMnist(delete_num, shuffle)
    elif args.dataset == 'covtype':
        return Load_covtype(delete_num, shuffle) 
    elif args.dataset == 'epsilon':
        return Load_epsilon(delete_num, shuffle) 
    elif args.dataset == 'gisette':
        return Load_gisette(delete_num, shuffle)
    elif args.dataset == 'ijcnn1':
        return Load_ijcnn1()
    elif args.dataset == 'higgs': 
        return Load_higgs(delete_num, shuffle)
    elif args.dataset == 'madelon':
        return Load_madelon(delete_num, shuffle)
    elif args.dataset == 'phishing':
        return Load_phishing(delete_num, shuffle)
    elif args.dataset == 'splice':
        return Load_splice(delete_num, shuffle)
    else:
        raise Exception('no such dataset, please recheck dataset name !')

if __name__ == '__main__':

    train_data, test_data, sequence = Load_binaryMnist(delete_num=0)

    train_data, test_data, sequence = Load_covtype(delete_num=0)

