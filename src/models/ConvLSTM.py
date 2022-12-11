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
from models import ConvLSTM, ConvolutionOverfit, ConvolutionOverfit3D
from utils import *
from runners import Trainer
from data import get_constant_data, get_half_half, get_train_test_dataloader
from torchinfo import summary

# path to the data on the vayne server
VAYNE_PATH = "/home/joe/ai-practicum/fmri-data/"


def main(device):
    random.seed(123)
    batch_size = 3
    # training_generator, test_generator = get_train_test_dataloader(
    #     (0.8, 0.2), batch_size)
    # data = get_constant_data()
    # data = get_half_half(8, VAYNE_PATH)
    data = get_half_half(8)
    # model = ConvLSTM(conv_kernel = 3, pool_kernel = 2, input_dim = 192, output_dim = 192)
    model = ConvolutionOverfit3D()
    # summary(model.to(0), (1,140, 48, 64, 64))

    # model = ConvLSTM2(input_dim = 128, output_dim = 128)
    # model = ConvolutionOverfit()

    # adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # sug = torch.optim.SGD(model.parameters(), lr=0.0001)

    # ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    # trainer = Trainer( model = model, optimizer = adam_optimizer, loss_fn = ce_loss, gpu_id = 0, save_interval = 1, metric_interval = 1, train_data = training_generator, validation_data = test_generator)
    # trainer = Trainer( model = model, optimizer = adam_optimizer, loss_fn = ce_loss, gpu_id = 0, save_interval = 1, metric_interval = 1, train_data = test_generator)
    # trainer = Trainer( model = model, optimizer = sug, loss_fn = ce_loss, gpu_id = 0, save_interval = 1, metric_interval = 1, train_data = data)
    trainer = Trainer(model=model, optimizer=adam_optimizer, loss_fn=bce_loss,
                      gpu_id=0, save_interval=1, metric_interval=1, train_data=data)

    # assert False
    s = datetime.now()
    print('Starting Training')
    num_epochs = 100
    trainer.train(num_epochs)
    print('Finished Training')
    f = datetime.now()
    print(f'Time to run {num_epochs} epochs: {f-s} (HH:MM:SS)')


if __name__ == "__main__":
    import sys
    device = 0
    main(device)
