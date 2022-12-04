import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os 
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

    def __init__(self, fmri_scan_ids: list, labels: dict):
        assert (len(fmri_scan_ids) == len(labels)), f'len(fmri_scan_ids) {len(fmri_scan_ids)} != len(labels) {len(labels)}'
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

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, output_dim, batch_first = True)
        self.fc = nn.Linear(output_dim, 2)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, input_tensor):
        # B, 140, 192]
        output, (h_n, c_n) = self.lstm(input_tensor)
        print(torch.squeeze(h_n, 0).shape)
        print(output[:,-1,:].shape)
        # assert( torch.equal(torch.squeeze(h_n, 0), output[:,-1,:]) )
        X = self.fc(torch.squeeze(h_n, 0))
        print(X)
        preds = self.softmax(X)
        print(preds)
        return preds
    
class ConvLSTM(nn.Module):
    def __init__(self, conv_kernel, pool_kernel, input_dim, output_dim):
        super(ConvLSTM, self).__init__()
        self.conv_kernel = conv_kernel
        self.pool_kernel = pool_kernel
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.convolution = Convolution( conv_kernel, pool_kernel)
        self.lstm = LSTM(input_dim, output_dim)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        X = self.convolution(input_tensor)
        X = self.lstm(X)
        return X

class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn,
        gpu_id: int,
        save_interval: int,
        train_data: DataLoader,
        validation_data: DataLoader = None,
        test_data: DataLoader = None
    ) -> None:
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.gpu_id = gpu_id
        self.save_interval = save_interval
        self.validation_data = validation_data
        self.test_data = test_data
    
    def _run_batch(self, batch_tensor: torch.tensor, batch_labels: torch.tensor):
        self.optimizer.zero_grad()
        precidted_output = self.model(batch_tensor)
        loss = self.loss_fn(predicted_output, batch_labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        for batch_tensor, batch_labels in self.train_data:
            batch_tensor = batch_tensor.to(self.gpu_id)
            # check batch labels type
            batch_labels = batch_labels.to(self.gpu_id)
            self._run_batch(batch_tensor, batch_labels)

    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, 'checkpoint_model.pt')
        print(f'\t Model Saved at Epoch {epoch}')

    def train(self, num_epochs: int):
        for epoch in range(1, num_epochs + 1):
            self._run_epoch(epoch)
            if self.save_interval > 0 and epoch % self.self.save_interval == 0:
                self._save_checkpoint(epoch)
            elif epoch == num_epochs:
                self._save_checkpoint(epoch)
        
        if self.validation_data != None:
            self.evaluate(self.train_data)
            self.evaluate(self.validation_data)

    def evaluate(self, dataloader: DataLoader):
        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            num_correct = 0
            total = 0
            num_batches = len(dataloader)

            for batch_tensor, batch_labels in dataloader:
                batch_tensor = batch.tensor.to(self.gpu_id)
                # check batch labels type
                batch_labels = batch_labels.to(self.gpu_id)
                predicted_output = self.model(batch_tensor)

                cumulative_loss += self.loss_fn(predicted_output, batch_labels)
                #assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                num_correct += (torch.round(predicted_output) == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total
            print(f'\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\tAccuracy: {accuracy} = {num_correct}/{total}')
            print()

        self.model.train()

def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs):
    print('starting training')
    curr_epoch = 1
    for epoch in range(epochs):
        print(f'Starting epoch: {curr_epoch}/{epochs}')
        epoch_start_time = time.time()
        # Training
        model.train()
        for local_batch, lintocal_labels in train_dataloader:
            print(local_batch.shape)
            print(local_labels)
            optimizer.zero_grad()
            predicted_output = model(local_batch)
            print(predicted_output.shape)
            print(local_labels.shape)
            assert (False)
            curr_loss = loss_fn(predicted_output, local_labels)
            curr_loss.backward()
            optimizer.step()

        # evaluate_model(train_dataloader, model, loss_fn)

        print(f'Time for epoch {curr_epoch}: {time.time() - epoch_start_time}')
        curr_epoch += 1

    print('finished training')


def main(device):
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


    training_set = FMRIDataset(train_scans, train_labels)
    test_set = FMRIDataset(test_scans, test_labels)

    training_generator = DataLoader(training_set, batch_size = 1)
    test_generator = DataLoader(test_set)

    model = Convolution(3, 2)
    conv_kernel = 3
    pool_kernel = 2
    input_dim = 192
    output_dim = 192
    model = ConvLSTM(conv_kernel, pool_kernel, input_dim, output_dim)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce_loss = nn.CrossEntropyLoss()

    trainer = Trainer( model = model, optimizer = adam_optimizer, loss_fn = ce_loss, gpu_id = 0, save_interval = 0, train_data  = training_generator, validation_data = test_generator)

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
    import sys
    device = 0 
    main(device)
