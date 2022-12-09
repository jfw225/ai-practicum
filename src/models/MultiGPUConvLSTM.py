import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os 
import pandas as pd
import pickle
import random
import itertools
import time
from datetime import datetime, timedelta
from torchmetrics import ROC
from torchmetrics.classification import BinaryROC
from matplotlib import pyplot as plt

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



def sequential_train_test_split(split, subject_scans_dict):
    assert (len(split) == 2)
    assert (sum(split) == 1)

    subject_to_num_scans = dict(zip(list(subject_scans_dict.keys()), map(
        lambda x: len(x), list(subject_scans_dict.values()))))
    num_scans = sum(list(subject_to_num_scans.values()))

    all_subjects = list(subject_scans_dict.keys())
    num_train_scans = 0
    train_subjects = []
    i = 0
    while num_train_scans/num_scans < split[0]:
        curr_subject = all_subjects[i]
        train_subjects.append(curr_subject)
        num_train_scans += subject_to_num_scans[curr_subject]
        i += 1

    train_scans = []
    for subject in train_subjects:
        train_scans += subject_scans_dict[subject]

    test_scans = []
    for subject in all_subjects[i:]:
        test_scans += subject_scans_dict[subject]

    return train_scans, test_scans

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

def get_FWHM_gaussian_kernel(fwhm):
    sigma = fwhm / np.sqrt(8 * np.log(2))
    ts = torch.arange(3.31*-3,3.31*4,3.31)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    k = gauss / gauss.sum()

    kernel_3d = torch.einsum('i,j,k->ijk', k, k, k)
    kernel_3d = kernel_3d / kernel_3d.sum()
    return  kernel_3d

def get_FWHM_gaussian_blur(t, kernel_3d):
    reshaped_t = t[:,None,:,:,:].float().to(0)
    reshaped_k = kernel_3d[None,None,:,:,:].to(0)

    # 7 = kernel_3d.shape[0]= len(k) = len(ts) = len(gauss)

    vol_3d = F.conv3d(reshaped_t, reshaped_k, stride=1, padding= 7 // 2)

    return torch.squeeze(vol_3d)

class FMRIDataset(Dataset):

    def __init__(self, fmri_scan_ids: list, labels: dict, normalize: bool, kernel_3d : torch.tensor):
        assert (len(fmri_scan_ids) == len(labels)), f'len(fmri_scan_ids) {len(fmri_scan_ids)} != len(labels) {len(labels)}'
        self.fmri_scan_ids = fmri_scan_ids
        self.labels = labels
        self.normalize = normalize
        if normalize:
            self.normalize_dict = FMRIDataset._get_normalize_dict(fmri_scan_ids)
        else:
            self.normalize_dict = None
        self.kernel_3d = kernel_3d

    @staticmethod
    def _get_normalize_dict(fmri_scan_ids):
        normalize_dict = dict()
        for fmri_scan_id in fmri_scan_ids:
            x = torch.load(f'/home/ai-prac/ai-practicum/fmri-data/torch-data/data/{fmri_scan_id}.pt')
            normalize_dict[fmri_scan_id] = transforms.Normalize( torch.mean(x.float()), torch.std(x.float()) )
            x = normalize_dict[fmri_scan_id](x.float())            
        return normalize_dict

    def __len__(self):
        return len(self.fmri_scan_ids)

    def __getitem__(self, fmri_scan_idx):
        fmri_scan_id = self.fmri_scan_ids[fmri_scan_idx]

        if self.normalize == False:
            x = torch.load(
                f'/home/ai-prac/ai-practicum/fmri-data/torch-data/data/{fmri_scan_id}.pt')
            # x = get_FWHM_gaussian_blur(x, self.kernel_3d)
            y = self.labels[fmri_scan_id]
        else:
            x = torch.load(
                f'/home/ai-prac/ai-practicum/fmri-data/torch-data/data/{fmri_scan_id}.pt')
            norm_func = self.normalize_dict[fmri_scan_id]
            x = norm_func(x.float())
            y = self.labels[fmri_scan_id]
 
        return x, y

class ConvolutionOverfit(nn.Module):
    def __init__(self):
        super(ConvolutionOverfit, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 8, (7,7,7), stride = 2)
        self.conv3d_2 = nn.Conv3d(8, 16, (5,5,5), stride = 2)
        self.conv3d_3 = nn.Conv3d(16, 32, (3,3,3), stride = 2)
        self.conv3d_4 = nn.Conv3d(32, 16, (3,3,3), stride = 2)
        self.fc1 = nn.Linear(140*64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.reshape = nn.Flatten(0, 1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = torch.sigmoid
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        X = self.reshape(input_tensor)
        X = X[:, None, :, :, :]

        X = self.conv3d_1(X)
        X = self.relu(X)
        X = self.conv3d_2(X)
        X = self.relu(X)
        # X = self.dropout(X)
        X = self.conv3d_3(X)
        X = self.relu(X)
        ### X = self.dropout(X)
        X = self.conv3d_4(X)
        # X = self.dropout(X)
        X = torch.flatten(X) # full flatten (not-batched)
        X = self.fc1(X)
        X = self.sigmoid(X)
        X = self.fc2(X)
        X = self.sigmoid(X)
        X = self.fc3(X)
        X = X[None, :]
        assert (1, 2) == X.shape
        return X

class Convolution2(nn.Module):
    def __init__(self):
        super(Convolution2, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 8, (7,7,7), stride = 2)
        self.conv3d_2 = nn.Conv3d(8, 16, (5,5,5), stride = 2)
        self.conv3d_3 = nn.Conv3d(16, 32, (3,3,3), stride = 2)
        self.fc1 = nn.Linear(4608, 128)

        self.reshape = nn.Flatten(0, 1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = torch.sigmoid
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        X = self.reshape(input_tensor)
        X = X[:, None, :, :, :]

        X = self.conv3d_1(X)
        X = self.relu(X)
        X = self.conv3d_2(X)
        X = self.relu(X)
        # X = self.dropout(X)
        X = self.conv3d_3(X)
        X = self.relu(X)
        # X = self.dropout(X)
        X = self.flatten(X)
        X = self.fc1(X)
        X = self.sigmoid(X)

        assert (B*num_slides, 128) == X.shape
        X = torch.reshape(X, (B, num_slides, 128))
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
 
        self.flatten = nn.Flatten() # batch flatten
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(576, 192)
        # self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        # print(f'1: {(B, num_slides, num_slices, h, w)}')

        X = self.reshape(input_tensor)
        # print(f'2: {X.shape}')
        X = X[:, None, :, :, :]
        # print(f'3: {X.shape}')
        # print(X.dtype)

        X = self.cov3d_1(X.float())
        # print(X.shape)
        ### X = self.batchnorm_1(X)
        X = self.relu(X)
        X = self.pooling3d_1(X)
        

        # print(X.shape)
        X = self.cov3d_2(X)
        # print(X.shape)
        ### X = self.batchnorm_2(X)
        X = self.relu(X)

        X = self.pooling3d_2(X)
        ### X = self.relu(X)
        # print(X.shape)

        X = self.cov3d_3(X)
        X = self.relu(X)
        # print(X.shape)
        ### X = self.batchnorm_3(X)
        # print(X.shape)
        X = self.cov3d_4(X)
        X = self.relu(X)
        # print(X.shape)

        ### X = self.batchnorm_4(X)
        X = self.pooling3d_3(X)
        # print(X.shape)

        X = self.cov3d_5(X)
        # print(X.shape)
        ### X = self.batchnorm_5(X)
        X = self.relu(X)
        # print(X.shape)
        # X = self.cov3d_6(X)
        # X = self.pooling3d_3(X)

        X = self.flatten(X)
        # print(X.shape)
        # X = self.dropout(X)
        X = self.fc1(X)
        # print(X.shape)

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
        # [B, 140, 192]
        output, (h_n, c_n) = self.lstm(input_tensor)
        # print(torch.squeeze(h_n, 0).shape)
        # print(output[:,-1,:].shape)
        # assert( torch.equal(torch.squeeze(h_n, 0), output[:,-1,:]) )
        X = self.fc(torch.squeeze(h_n, 0))
        # print(X)
        # preds = self.softmax(X)
        # print(preds)
        return X
    
class ConvLSTM(nn.Module):
    def __init__(self, conv_kernel, pool_kernel, input_dim, output_dim):
        super(ConvLSTM, self).__init__()
        self.conv_kernel = conv_kernel
        self.pool_kernel = pool_kernel
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.dropout = nn.Dropout(0.2)

        self.convolution = Convolution( conv_kernel, pool_kernel)
        self.lstm = LSTM(input_dim, output_dim)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        # X = self.dropout(input_tensor)
        X = self.convolution(input_tensor)
        # print(X)
        X = self.lstm(X)
        return X

class ConvLSTM2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvLSTM2, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.dropout = nn.Dropout(0.2)

        self.convolution = Convolution2()
        self.lstm = LSTM(input_dim, output_dim)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        # X = self.dropout(input_tensor)
        X = self.convolution(input_tensor)
        # print(X)
        X = self.lstm(X)
        return X

def ddp_setup(rank, world_size):
  '''
  Args: 
      rank: Unique identifier of each process
      world_size: Total number of processes
  '''
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
  init_process_group(backend = 'nccl', rank=rank, world_size= world_size)


#TODO: self.model() ==> self.model.module()
# refactor train()

class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn,
        gpu_id: int,
        save_interval: int,
        metric_interval: int,
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
        self.metric_interval = metric_interval
        self.validation_data = validation_data
        self.test_data = test_data
        self.model = DDP(self.model, device_ids = [self.gpu_id], find_unused_parameters=True)

    
    def _run_batch(self, batch_tensor: torch.tensor, batch_labels: torch.tensor):
        self.optimizer.zero_grad()
        predicted_output = self.model(batch_tensor)
        loss = self.loss_fn(predicted_output, batch_labels)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        self.model.train()
        print(f'\t[GPU {self.gpu_id}] Epoch {epoch}')
        i = 1
        ########### Did not fix bug #############
        # self.train_data.sampler.set_epoch(epoch)
        ########################################
        for batch_tensor, batch_labels in self.train_data:
            # print(f'\t{i}/{len(self.train_data)}')
            i += 1
            batch_tensor = batch_tensor.to(self.gpu_id)
            # check batch labels type
            batch_labels = batch_labels.to(self.gpu_id)
            self._run_batch(batch_tensor, batch_labels)

    def _save_checkpoint(self, epoch: int):
        checkpoint = self.model.module.state_dict()
        torch.save(checkpoint, 'checkpoint_model.pt')
        print(f'\tModel Saved at Epoch {epoch}')

    def train(self, num_epochs: int):
        # output last if self.metric_interval is less than 1 always
        # output last if num_epochs % self.metric_interval != 0

        for epoch in range(1, num_epochs + 1):
            self._run_epoch(epoch)
            
            if self.gpu_id == 0 and (self.save_interval > 0 and epoch % self.save_interval == 0):
                self._save_checkpoint(epoch)
            elif self.gpu_id == 0 and epoch == num_epochs:  ## save last model
                self._save_checkpoint(epoch)

            # if self.gpu_id == 0 and (self.metric_interval > 0 and epoch % self.metric_interval == 0):
            #     self.evaluate(self.train_data, sv_roc = True)
            #     if self.validation_data != None:
            #         self.evaluate(self.validation_data)
            # elif self.gpu_id == 0 and epoch == num_epochs:  ## Evaluate final model 
            #     self.evaluate(self.train_data, sv_roc = True)
            #     if self.validation_data != None:
            #         self.evaluate(self.validation_data)

    def evaluate(self, dataloader: DataLoader, sv_roc = False):
        with torch.no_grad():
            self.model.eval()
            cumulative_loss = 0
            num_correct = 0
            total = 0
            num_batches = len(dataloader)
            all_preds = torch.tensor([]).to(self.gpu_id)
            all_labels = torch.tensor([]).to(self.gpu_id)

            for batch_tensor, batch_labels in dataloader:
                batch_tensor = batch_tensor.to(self.gpu_id)
                # check batch labels type
                batch_labels = batch_labels.to(self.gpu_id)
                predicted_output = self.model(batch_tensor)
                cumulative_loss += self.loss_fn(predicted_output, batch_labels)
                if sv_roc:
                    softmax = nn.Softmax(dim = 1)
                    all_preds = torch.cat( (all_preds, (softmax(predicted_output)[:,1])) )
                    all_labels = torch.cat( (all_labels, batch_labels) )

                #assuming decision boundary to be 0.5
                total += batch_labels.size(0)
                num_correct += (torch.argmax(predicted_output) == batch_labels).sum().item()

            loss = cumulative_loss/num_batches
            accuracy = num_correct/total
            print(f'\t\tLoss: {loss} = {cumulative_loss}/{num_batches}')
            print(f'\t\tAccuracy: {accuracy} = {num_correct}/{total}')
            if sv_roc:
                Trainer.save_roc(all_preds, all_labels)
            print()

        self.model.train()

    @staticmethod
    def save_roc( all_preds, all_labels):
        # print(all_preds)
        # print(all_labels)
        roc = ROC(task="binary", thresholds = 20)
        roc = BinaryROC(thresholds = 20)
        all_preds = all_preds.cpu()
        all_labels = all_labels.cpu().int()
        # print("##############")
        # print(all_preds)
        # print(all_labels)
        # print("##############")
        fpr, tpr, thresholds = roc(all_preds, all_labels)
        plt.plot([0,1],[0,1], linestyle='dashed')
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('ROC.png')

def get_train_test_dataset(split : tuple):
        assert( sum(split) == 1)
        assert( len(split) == 2 ) 
        # Create train and test split
        PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data'
        subject_scans_dict = pickle.load(
            open(f'{PATH}/subject-scans-dict.pickle', 'rb'))
        # train_scans, test_scans = train_test_split(split, subject_scans_dict)
        train_scans, test_scans = sequential_train_test_split(split, subject_scans_dict)

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

        kernel_3d = get_FWHM_gaussian_kernel(6)
        assert( 7 == kernel_3d.shape[0] )

        training_set = FMRIDataset(train_scans, train_labels, normalize = False, kernel_3d = kernel_3d)
        print(f'Num Train 0 (CN): {list(train_labels.values()).count(0)}')
        print(f'Num Train 1 (AD): {list(train_labels.values()).count(1)}')

        test_set = FMRIDataset(test_scans, test_labels, normalize = False, kernel_3d = kernel_3d)
        print(f'Num Test 0 (CN): {list(test_labels.values()).count(0)}')
        print(f'Num Test 1 (AD): {list(test_labels.values()).count(1)}')

        return training_set, test_set


def get_train_test_dataloader( split: tuple, batch_size):
    training_set, test_set = get_train_test_dataset(split)

    training_generator = DataLoader(
                            training_set, 
                            batch_size = batch_size,
                            pin_memory = True, ## TODO: Does nothing since get_FWHM_gaussian_blur() calls .to(0)
                            shuffle = False, # False since since sampler does shuffling
                            sampler = DistributedSampler(training_set))

    test_generator = DataLoader(
                        test_set, 
                        batch_size = 1,
                        pin_memory = True, ## TODO: Does nothing since get_FWHM_gaussian_blur() calls .to(0)
                        shuffle = False, # False since sampler does shuffling
                        sampler = DistributedSampler(test_set))

    return training_generator, test_generator

def load_and_test_model(untrained_model,data_generator):
    
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce_loss = nn.CrossEntropyLoss()
    rank = 0
        
    untrained_model.load_state_dict(torch.load('checkpoint_model.pt'))
    untrained_model.evaluate( data_generator, sv_roc = False)

    trainer = Trainer( model = untrained_model, optimizer = adam_optimizer, loss_fn = ce_loss, gpu_id = rank, save_interval = 1, metric_interval = 1, train_data = training_generator)



def main(rank: int, world_size: int):
    random.seed(123)
    ddp_setup(rank, world_size)
    batch_size = 1
    training_generator, test_generator = get_train_test_dataloader((0.8, 0.2), batch_size)

    model = ConvLSTM(conv_kernel = 3, pool_kernel = 2, input_dim = 192, output_dim = 192)
    ## model = ConvLSTM2(input_dim = 128, output_dim = 128)
    ## model = ConvolutionOverfit()


    ## adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ce_loss = nn.CrossEntropyLoss()

    trainer = Trainer( model = model, optimizer = adam_optimizer, loss_fn = ce_loss, gpu_id = rank, save_interval = 1, metric_interval = 1, train_data = training_generator, validation_data = test_generator)
    ## assert False
    s = datetime.now()
    print('Starting Training')
    num_epochs = 10
    trainer.train(num_epochs)
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
