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
from data import FMRIDataset


def get_FWHM_gaussian_kernel(fwhm):
    sigma = fwhm / np.sqrt(8 * np.log(2))
    ts = torch.arange(3.31*-3,3.31*4,3.31)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    k = gauss / gauss.sum()

    kernel_3d = torch.einsum('i,j,k->ijk', k, k, k)
    kernel_3d = kernel_3d / kernel_3d.sum()
    return  kernel_3d

def get_FWHM_gaussian_blur(t, kernel_3d):
    # reshaped_t = t[:,None,:,:,:].float().to(0)
    # reshaped_k = kernel_3d[None,None,:,:,:].to(0)
    reshaped_t = t[:,None,:,:,:].float()
    reshaped_k = kernel_3d[None,None,:,:,:]

    # 7 = kernel_3d.shape[0]= len(k) = len(ts) = len(gauss)

    vol_3d = F.conv3d(reshaped_t, reshaped_k, stride=1, padding= 7 // 2)

    return torch.squeeze(vol_3d)


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


        #################
        # t = torch.rand((140,48,64,64))

        # a = time.time()
        # for i in range(100):
        #     get_FWHM_gaussian_blur(t, kernel_3d)
        # b = time.time()
        # print(f'No gpu time {b - a}')

        # a = time.time()
        # for i in range(100):
        #     get_FWHM_gaussian_blur(t.to(0), kernel_3d.to(0))
        # b = time.time()
        # print(f'No gpu time {b - a}')

        # assert False
        #################

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
                            pin_memory = False, ## TODO: Does nothing since get_FWHM_gaussian_blur() calls .to(0)
                            shuffle = True)

    test_generator = DataLoader(
                        test_set, 
                        batch_size = 1,
                        pin_memory = False, ## TODO: Does nothing since get_FWHM_gaussian_blur() calls .to(0)
                        shuffle = True)

    return training_generator, test_generator


