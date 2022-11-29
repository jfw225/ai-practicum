import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import random
import itertools
import time


def train_test_split(split, subject_scans_dict):
    assert (len(split) == 2)
    assert (sum(split) == 1)

    subject_to_num_scans = dict(zip(list(subject_scans_dict.keys()), map(
        lambda x: len(x), list(subject_scans_dict.values()))))
    num_scans = sum(list(subject_to_num_scans.values()))

    all_subjects = set(list(subject_scans_dict.keys()))
    num_train_scans = 0
    train_subjects = set()
    while num_train_scans/num_scans < split[0]:
        curr_subject = random.choice(list(all_subjects - train_subjects))
        train_subjects.add(curr_subject)
        num_train_scans += subject_to_num_scans[curr_subject]

    # print(num_train_scans/num_scans)
    test_subjects = all_subjects - train_subjects

    # assert(len(test_subjects) + len(train_subjects) == len(subject_scans_dict))

    train_scans = []
    for subject in train_subjects:
        train_scans += subject_scans_dict[subject]

    test_scans = []
    for subject in test_subjects:
        test_scans += subject_scans_dict[subject]

    # print(len(test_scans) + len(train_scans))

    # assert(len(set(test_scans).union(set(train_scans))) == 302)
    return train_scans, test_scans


class FMRIDataset(Dataset):

    def __init__(self, fmri_scan_ids, labels):
        self.fmri_scan_ids = fmri_scan_ids
        self.labels = labels

    def __len__(self):
        return len(self.fmri_scan_ids)

    def __getitem__(self, fmri_scan_idx):
        fmri_scan_id = self.fmri_scan_ids[fmri_scan_idx]

        x = torch.load(
            f'/home/ai-prac/ai-practicum/fmri-data/torch-data/data/{fmri_scan_id}.pt')
        y = self.labels[fmri_scan_id]
        return x, y

        return train_scans, test_scans


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.softmax = torch.Softmax()
        self.convolution = Convolution(input_dim)
        self.lstm = LSTM(hidden_dim, out_dim)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        X = self.convolution(input_tensor)
        X = self.lstm(X)
        return X


class Convolution(nn.Module):
    def __init__(self, input_dim, conv_kernel, pool_kernel):
        self.input_dim = input_dim
        # (B, num_slides, num_slices, h, w) = input_dim
        self.conv_kernel = 3
        self.pool_kernel = 2
        self.reshape = nn.Flatten(0, 1)

        self.cov3d_1 = nn.Conv3d(1, 32, conv_kernel)
        self.batchnorm_1 = nn.BatchNorm3d(32)
        self.pooling3d_1 = nn.MaxPool3d(pool_kernel)

        self.cov3d_2 = nn.Conv3d(32, 64, conv_kernel)
        self.batchnorm_2 = nn.BatchNorm3d(64)

        self.pooling3d_2 = nn.MaxPool3d(pool_kernel)

        self.cov3d_3 = nn.Conv3d(64, 128, conv_kernel)
        self.batchnorm_3 = nn.BatchNorm3d(128)
        self.cov3d_4 = nn.Conv3d(128, 128, conv_kernel)
        self.batchnorm_4 = nn.BatchNorm3d(128)
        self.pooling3d_3 = nn.MaxPool3d(pool_kernel)

        self.cov3d_5 = nn.Conv3d(128, 256, conv_kernel)
        self.batchnorm_5 = nn.BatchNorm3d(256)
        self.cov3d_6 = nn.Conv3d(256, 256, conv_kernel)
        self.pooling3d_3 = nn.MaxPool3d(pool_kernel)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape

        X = self.reshape(input_tensor)
        X = X[:, None, :, :, :]

        X = self.cov3d_1(input_tensor)
        X = self.batchnorm_1(X)
        X = self.pooling3d_1(X)

        X = self.cov3d_2(X)
        X = self.batchnorm_2(X)

        X = self.pooling3d_2(X)

        X = self.cov3d_3(X)
        X = self.batchnorm_3(X)
        X = self.cov3d_4(X)

        X = self.batchnorm_4(X)
        X = self.pooling3d_3(X)

        X = self.cov3d_5(X)
        X = self.batchnorm_5(X)
        X = self.cov3d_6(X)
        X = self.pooling3d_3(X)

        X = self.dropout(X)
        X = self.fc1(X)
        X = self.relu(X)

        assert (B*num_slides, 256) == X.shape
        X = torch.reshape(X, (B, num_slides, 256))

        return X


class LSTM():
    pass


def main():
    # Create custom dataset class

    # Create train and test split
    PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data'
    subject_scans_dict = pickle.load(
        open(f'{PATH}/subject-scans-dict.pickle', 'rb'))
    train_scans, test_scans = train_test_split((.8, .2), subject_scans_dict)

    # Find train and test labels

    PATH = '/home/ai-prac/ai-practicum/fmri-data/'
    fmri_df = pd.read_csv(f'{PATH}/fMRI_summary.csv')
    fmri_df = fmri_df[fmri_df['Description'] == 'Resting State fMRI']
    fmri_df = fmri_df[(fmri_df['Research Group'] == 'AD') |
                      (fmri_df['Research Group'] == 'CN')]
    keys = fmri_df["Image ID"].map(lambda x: f'I{x}')
    str_label_to_int = {'CN': 0, 'AD': 1}
    values = list(
        map(lambda x: str_label_to_int[x], fmri_df["Research Group"]))
    all_labels = dict(zip(keys, values))
    train_labels = {key: all_labels[key] for key in train_scans}
    test_labels = {key: all_labels[key] for key in test_scans}

    # print(train_scans)
    # print(test_scans)
    # print(train_labels)
    # print(test_labels)

    training_set = FMRIDataset(train_scans, train_labels)
    test_set = FMRIDataset(test_scans, test_labels)

    training_generator = DataLoader(training_set)
    test_generator = DataLoader(test_set)

    i = 1
    print('starting training')
    for epoch in range(1):
        # Training
        for local_batch, local_labels in training_generator:
            print(i)
            i += 1
            pass
            # Transfer to GPU
            # local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    test_labels = dict()
    str_label_to_int = {'CN': 0, 'AD': 1}
    for fmri in train_data:
        str_label = fmri_df[fmri_df["Image ID"] == fmri]['Research Group']
        print(str_label)
        int_label = str_label_to_int[str_label]
        test_labels[fmri] = int_label

    # print(train_scans)
    # print(test_scans)
    # print(train_labels)
    # print(test_labels)

        print(len(fmri_df))
    training_set = FMRIDataset(train_scans, train_labels)
    test_set = FMRIDataset(test_scans, test_labels)

    # Create data loader
    training_generator = DataLoader(training_set)
    test_generator = DataLoader(test_set)

    i = 1
    print('starting training')
    for epoch in range(1):
        # Training
        for local_batch, local_labels in training_generator:
            print(i)
            i += 1
            pass
            # Transfer to GPU
            # local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    print('finished training')

    # Create data loader

    # Implementation of model


    # Train/Evaluate functions
if __name__ == "__main__":
    main()
