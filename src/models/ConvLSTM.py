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

    test_subjects = all_subjects - train_subjects

    train_scans = []
    for subject in train_subjects:
        train_scans += subject_scans_dict[subject]

    test_scans = []
    for subject in test_subjects:
        test_scans += subject_scans_dict[subject]

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


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(ConvLSTM, self).__init__()
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
    def __init__(self, conv_kernel, pool_kernel):
        super(Convolution, self).__init__()
        # (B, num_slides, num_slices, h, w) = input_dim
        self.conv_kernel = 3
        self.pool_kernel = 2

        self.reshape = nn.Flatten(0, 1)

        self.cov3d_1 = nn.Conv3d(1, 8, conv_kernel)
        self.batchnorm_1 = nn.BatchNorm3d(8)
        self.pooling3d_1 = nn.MaxPool3d(pool_kernel)

        self.cov3d_2 = nn.Conv3d(8, 16, conv_kernel)
        self.batchnorm_2 = nn.BatchNorm3d(16)

        self.pooling3d_2 = nn.MaxPool3d(pool_kernel)

        self.cov3d_3 = nn.Conv3d(16, 32, conv_kernel)
        self.batchnorm_3 = nn.BatchNorm3d(32)
        self.cov3d_4 = nn.Conv3d(32, 32, conv_kernel)
        self.batchnorm_4 = nn.BatchNorm3d(32)
        self.pooling3d_3 = nn.MaxPool3d(pool_kernel)

        self.cov3d_5 = nn.Conv3d(32, 64, conv_kernel)
        self.batchnorm_5 = nn.BatchNorm3d(64)

        # self.cov3d_5 = nn.Conv3d(128, 256, conv_kernel)
        # self.batchnorm_5 = nn.BatchNorm3d(256)
        # self.cov3d_6 = nn.Conv3d(256, 256, conv_kernel)
        # self.pooling3d_3 = nn.MaxPool3d(pool_kernel)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(576, 192)
        # self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        print(f'1: {(B, num_slides, num_slices, h, w)}')

        X = self.reshape(input_tensor)
        print(f'2: {X.shape}')
        X = X[:, None, :, :, :]
        print(f'3: {X.shape}')
        print(X.dtype)

        X = self.cov3d_1(X.float())
        print(X.shape)
        X = self.batchnorm_1(X)
        X = self.pooling3d_1(X)

        print(X.shape)
        X = self.cov3d_2(X)
        print(X.shape)
        X = self.batchnorm_2(X)

        X = self.pooling3d_2(X)
        print(X.shape)

        X = self.cov3d_3(X)
        print(X.shape)
        X = self.batchnorm_3(X)
        print(X.shape)
        X = self.cov3d_4(X)
        print(X.shape)

        X = self.batchnorm_4(X)
        X = self.pooling3d_3(X)
        print(X.shape)

        X = self.cov3d_5(X)
        print(X.shape)
        X = self.batchnorm_5(X)
        print(X.shape)
        # X = self.cov3d_6(X)
        # X = self.pooling3d_3(X)

        X = self.flatten(X)
        print(X.shape)
        X = self.dropout(X)
        X = self.fc1(X)
        print(X.shape)
        X = self.relu(X)

        assert (B*num_slides, 192) == X.shape
        X = torch.reshape(X, (B, num_slides, 192))
        # X = torch.reshape(X, (B, num_slides, 256))

        return X


class LSTM():
    pass


def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs):
    print('starting training')
    curr_epoch = 1
    for epoch in range(epochs):
        print(f'Starting epoch: {curr_epoch}/{epochs}')
        epoch_start_time = time.time()
        # Training
        model.train()
        for local_batch, local_labels in train_dataloader:
            print(local_batch.shape)
            print(local_labels)
            optimizer.zero_grad()
            predicted_output = model(local_batch)
            print(predicted_output.shape)
            assert (False)
            curr_loss = loss_fn(predicted_output, local_labels)
            curr_loss.backward()
            optimizer.step()

        # evaluate_model(train_dataloader, model, loss_fn)

        print(f'Time for epoch {curr_epoch}: {time.time() - epoch_start_time}')
        curr_epoch += 1

    print('finished training')


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

    model = Convolution(3, 2)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, training_generator, test_generator,
          loss_fn=0, optimizer=adam_optimizer, epochs=1)

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


if __name__ == "__main__":
    main()
