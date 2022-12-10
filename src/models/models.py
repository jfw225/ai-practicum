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
        X = torch.reshape(X, (B, 140, 16, 1, 2, 2) )
        X = torch.flatten(X, 1) # (B, 140, 16*1*2*2)     
        X = self.fc1(X)
        X = self.sigmoid(X)
        X = self.fc2(X)
        X = self.sigmoid(X)
        X = self.fc3(X)
        X = X[None, :]
        X = X.reshape(-1, 2)
        assert (B, 2) == X.shape, f"X.shape = {X.shape}"
        return X

class ConvolutionOverfit3D(nn.Module):
    def __init__(self):
        super().__init__()
        # super(ConvolutionOverfit, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 8, (7,7,7), stride = 2)
        self.conv3d_2 = nn.Conv3d(8, 16, (5,5,5), stride = 2)
        self.conv3d_3 = nn.Conv3d(16, 32, (3,3,3), stride = 2)
        self.conv3d_4 = nn.Conv3d(32, 16, (3,3,3), stride = 2)
        self.fc1 = nn.Linear(1*64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)

        self.reshape = nn.Flatten(0, 1)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.sigmoid = torch.sigmoid
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor):
        (B, num_slides, num_slices, h, w) = input_tensor.shape
        print(input_tensor.shape)
        # X = self.reshape(input_tensor)
        # X = X[:, None, :, :, :]
        X = input_tensor[:, -1, :, :, :]
        X = X[:, None, :, :, :]
        print(X.shape)

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
        X = torch.reshape(X, (B, 1, 16, 1, 2, 2) )
        X = torch.flatten(X, 1) # (B, 140, 16*1*2*2)     
        X = self.fc1(X)
        X = self.sigmoid(X)
        X = self.fc2(X)
        X = self.sigmoid(X)
        X = self.fc3(X)
        X = X[None, :]
        X = X.reshape(-1, 2)
        assert (B, 2) == X.shape, f"X.shape = {X.shape}"
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
