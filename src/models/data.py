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
from utils import get_FWHM_gaussian_blur, get_FWHM_gaussian_kernel
from typing import Tuple

# the default path for jacob
DEFAULT_DATA_PATH = "/home/ai-prac/ai-practicum/fmri-data/"


class DataSummary:
    def __init__(self, data_path: str = None):
        # set the data path if it is not given
        data_path = data_path or DEFAULT_DATA_PATH

        # load the summary df
        self.fmri_df = DataSummary.load_summary(data_path)

        # create a mapping from subject id to image id
        self._label_map = self.create_label_map()

    def create_label_map(self):
        """
        Creates the mapping from scan IDs to labels.
        """

        keys = self.fmri_df["Image ID"].map(lambda x: f'I{x}')

        str_label_to_int = {'CN': 0, 'AD': 1}

        values = list(
            map(lambda x: str_label_to_int[x], self.fmri_df["Research Group"]))

        return dict(zip(keys, values))

    def get_label(self, scan_id):
        """
        Returns the label for a given scan ID.
        """

        return self._label_map[scan_id]

    @staticmethod
    def load_summary(data_path):
        """
        Loads the summary csv file.
        """

        # assert that the data path is a string
        assert isinstance(
            data_path, str), f"Data path must be a string: {data_path}!"

        fmri_df = pd.read_csv(os.path.join(data_path, "fMRI_summary.csv"))
        fmri_df = fmri_df[fmri_df['Description'] == 'Resting State fMRI']
        fmri_df = fmri_df[(fmri_df['Research Group'] == 'AD') |
                          (fmri_df['Research Group'] == 'CN')]

        return fmri_df


class FMRIDataset(Dataset):

    def __init__(self, fmri_scan_ids: list, labels: dict, normalize: bool, blur_fwhm: int, data_path: str = None):
        assert (len(fmri_scan_ids) == len(
            labels)), f'len(fmri_scan_ids) {len(fmri_scan_ids)} != len(labels) {len(labels)}'
        self.data_path = os.path.join(
            data_path or DEFAULT_DATA_PATH, "torch-data")
        self.fmri_scan_ids = fmri_scan_ids
        self.labels = labels
        self.normalize = normalize
        self.blur_fwhm = blur_fwhm
        self.kernel_3d = get_FWHM_gaussian_kernel(
            blur_fwhm) if self.should_blur else None
        self.mean, self.std = self._sample_mean_and_std()

    @property
    def should_blur(self):
        """
        Returns `True` if we are applying blur and `False` otherwise.
        """

        return bool(self.blur_fwhm)

    def _sample_mean_and_std(self):
        """
        Returns the sample mean and variance
        """
        total_sum, total_sum_of_squares, N = 0.0, 0.0, 0.0

        for fmri_scan_id in self.fmri_scan_ids:
            tensor = torch.load(self.scan_path(fmri_scan_id)).float()
            N += tensor.numel()
            total_sum += tensor.sum()
            total_sum_of_squares += (tensor ** 2).sum()

        mean = total_sum / N
        variance = (total_sum_of_squares - total_sum**2/N) / (N-1)
        std = torch.sqrt(variance)
        return mean, std

    def scan_path(self, fmri_scan_id):
        """
        Returns the path to the scan data for a given scan ID.
        """

        return os.path.join(self.data_path, "data", f'{fmri_scan_id}.pt')

    def __len__(self):
        return len(self.fmri_scan_ids)

    def __getitem__(self, fmri_scan_idx):
        fmri_scan_id = self.fmri_scan_ids[fmri_scan_idx]

        x = torch.load(self.scan_path(fmri_scan_id))
        if self.normalize == True:
            x = (x - self.mean)/self.std
        if self.should_blur == True:
            x = get_FWHM_gaussian_blur(x, self.kernel_3d)

        y = self.labels[fmri_scan_id]

        return x.float(), y


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


def get_train_test_dataset(split: tuple, data_path=None) -> Tuple[FMRIDataset, FMRIDataset]:
    assert (sum(split) == 1)
    assert (len(split) == 2)
    # Create train and test split
    PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data'
    subject_scans_dict = pickle.load(
        open(f'{PATH}/subject-scans-dict.pickle', 'rb'))
    # train_scans, test_scans = train_test_split(split, subject_scans_dict)
    train_scans, test_scans = sequential_train_test_split(
        split, subject_scans_dict)

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

    training_set = FMRIDataset(
        train_scans, train_labels, normalize=True, blur_fwhm=6, data_path=data_path)

    print(f'Num Train 0 (CN): {list(train_labels.values()).count(0)}')
    print(f'Num Train 1 (AD): {list(train_labels.values()).count(1)}')
    print(f'Normalize: {True}')
    print(f'FWHM: {6}')

    test_set = FMRIDataset(test_scans, test_labels,
                           normalize=True, blur_fwhm=6, data_path=data_path)

    print(f'Num Test 0 (CN): {list(test_labels.values()).count(0)}')
    print(f'Num Test 1 (AD): {list(test_labels.values()).count(1)}')
    print(f'Normalize: {True}')
    print(f'FWHM: {6}')

    return training_set, test_set


def get_train_test_dataloader(split: tuple, batch_size, data_path=None) -> Tuple[DataLoader, DataLoader]:
    training_set, test_set = get_train_test_dataset(split)

    training_generator = DataLoader(
        training_set,
        batch_size=batch_size,
        pin_memory=False,  # TODO: Does nothing since get_FWHM_gaussian_blur() calls .to(0)
        shuffle=True)

    test_generator = DataLoader(
        test_set,
        batch_size=1,
        pin_memory=False,  # TODO: Does nothing since get_FWHM_gaussian_blur() calls .to(0)
        shuffle=True)

    return training_generator, test_generator


def get_constant_data() -> DataLoader:
    """
    Returns a data generator that is guaranteed to be the same data each time 
    this function is called.
    """

    # first two should be 0, last two should be 1
    scan_ids = ['I238623', 'I243902', 'I272407', 'I264986']

    # create the data summary
    data_summary = DataSummary()

    # get the labels
    labels = {scan_id: data_summary.get_label(scan_id) for scan_id in scan_ids}

    # ensure that we have the correct number of labels
    assert (len(labels) == len(scan_ids)
            ), f'len(labels) {len(labels)} != len(scan_ids) {len(scan_ids)}'

    # ensure that the labels are what we expect
    assert (labels['I238623'] ==
            0), f'labels["I238623"] {labels["I238623"]} != 0'
    assert (labels['I243902'] ==
            0), f'labels["I243902"] {labels["I243902"]} != 0'
    assert (labels['I272407'] ==
            1), f'labels["I272407"] {labels["I272407"]} != 1'
    assert (labels['I264986'] ==
            1), f'labels["I264986"] {labels["I264986"]} != 1'

    # create the data set
    data_set = FMRIDataset(
        scan_ids, labels, normalize=False, blur_fwhm=0)

    # create the data loader
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

    return data_loader


def get_half_half(n: int, data_path=None) -> DataLoader:
    """
    Returns a data set whose labels have `n` zeros and `n` ones.
    """

    # create the data summary
    data_summary = DataSummary(data_path=data_path)

    # get the mapping from scan id to label
    label_map = data_summary._label_map

    # get the scan ids for n zeros and n ones
    zero_scan_ids = [scan_id for scan_id,
                     label in label_map.items() if label == 0][:n]
    one_scan_ids = [scan_id for scan_id,
                    label in label_map.items() if label == 1][:n]

    # concatenate the scan ids
    scan_ids = zero_scan_ids + one_scan_ids

    # get the labels
    labels = {scan_id: data_summary.get_label(scan_id) for scan_id in scan_ids}

    # ensure that each scan id in `zero_scan_ids` has a label of 0
    for scan_id in zero_scan_ids:
        assert (labels[scan_id] ==
                0), f'labels[{scan_id}] {labels[scan_id]} != 0'

    # ensure that each scan id in `one_scan_ids` has a label of 1
    for scan_id in one_scan_ids:
        assert (labels[scan_id] ==
                1), f'labels[{scan_id}] {labels[scan_id]} != 1'

    # create the data set
    data_set = FMRIDataset(scan_ids, labels, normalize=True,
                           blur_fwhm=0, data_path=data_path)

    # create the data loader
    data_loader = DataLoader(data_set, batch_size=4, shuffle=False)

    return data_loader
