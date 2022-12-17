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


def get_FWHM_gaussian_kernel(fwhm):
    sigma = fwhm / np.sqrt(8 * np.log(2))
    ts = torch.arange(3.31*-3, 3.31*4, 3.31)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    k = gauss / gauss.sum()

    kernel_3d = torch.einsum('i,j,k->ijk', k, k, k)
    kernel_3d = kernel_3d / kernel_3d.sum()
    return kernel_3d


def get_FWHM_gaussian_blur(t, kernel_3d):
    # reshaped_t = t[:, None, :, :, :].float()
    # reshaped_k = kernel_3d[None, None, :, :, :]
    reshaped_t = t[:, None, :, :, :].float().to('cuda')
    reshaped_k = kernel_3d[None, None, :, :, :].to('cuda')

    # 7 = kernel_3d.shape[0]= len(k) = len(ts) = len(gauss)

    vol_3d = F.conv3d(reshaped_t, reshaped_k, stride=1,
                      padding=7 // 2)

    return torch.squeeze(vol_3d)
