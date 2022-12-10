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

# the default path for jacob
DEFAULT_DATA_PATH = "/home/ai-prac/ai-practicum/fmri-data/"

class DataSummary:
    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
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

        fmri_df = pd.read_csv(os.path.join(data_path, "fMRI_summary.csv"))
        fmri_df = fmri_df[fmri_df['Description'] == 'Resting State fMRI']
        fmri_df = fmri_df[(fmri_df['Research Group'] == 'AD') |
                        (fmri_df['Research Group'] == 'CN')]


        return fmri_df

class FMRIDataset(Dataset):

    def __init__(self, fmri_scan_ids: list, labels: dict, normalize: bool, kernel_3d : torch.tensor, data_path:str = None):
        assert (len(fmri_scan_ids) == len(labels)), f'len(fmri_scan_ids) {len(fmri_scan_ids)} != len(labels) {len(labels)}'
        self.fmri_scan_ids = fmri_scan_ids
        self.labels = labels
        self.normalize = normalize
        if normalize:
            self.normalize_dict = self._get_normalize_dict(fmri_scan_ids)
        else:
            self.normalize_dict = None

        # gaussian 3d conv
        self.kernel_3d = kernel_3d

        self.data_path = os.path.join(data_path or DEFAULT_DATA_PATH, "torch-data")


    def scan_path(self, fmri_scan_id):
        """
        Returns the path to the scan data for a given scan ID.
        """

        return os.path.join(self.data_path, "data", f'{fmri_scan_id}.pt')

    def _get_normalize_dict(self, fmri_scan_ids):
        normalize_dict = dict()
        for fmri_scan_id in fmri_scan_ids:
            x = torch.load(self.scan_path(fmri_scan_id))
            normalize_dict[fmri_scan_id] = transforms.Normalize( torch.mean(x.float()), torch.std(x.float()) )
            x = normalize_dict[fmri_scan_id](x.float())            
        return normalize_dict

    def __len__(self):
        return len(self.fmri_scan_ids)

    def __getitem__(self, fmri_scan_idx):
        fmri_scan_id = self.fmri_scan_ids[fmri_scan_idx]

        if self.normalize == False:
            x = torch.load(self.scan_path(fmri_scan_id))
            # x = get_FWHM_gaussian_blur(x, self.kernel_3d)
            y = self.labels[fmri_scan_id]
        else:
            x = torch.load(self.scan_path(fmri_scan_id))
            norm_func = self.normalize_dict[fmri_scan_id]
            x = norm_func(x.float())
            y = self.labels[fmri_scan_id]
 
        return x.float(), y


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
    assert (len(labels) == len(scan_ids)), f'len(labels) {len(labels)} != len(scan_ids) {len(scan_ids)}'

    # ensure that the labels are what we expect
    assert (labels['I238623'] == 0), f'labels["I238623"] {labels["I238623"]} != 0'
    assert (labels['I243902'] == 0), f'labels["I243902"] {labels["I243902"]} != 0'
    assert (labels['I272407'] == 1), f'labels["I272407"] {labels["I272407"]} != 1'
    assert (labels['I264986'] == 1), f'labels["I264986"] {labels["I264986"]} != 1'
    
    # create the data set
    data_set= FMRIDataset(scan_ids, labels, normalize=False, kernel_3d=None)

    # create the data loader
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

    return data_loader

def get_half_half(n: int, data_path=None)-> DataLoader:
    """
    Returns a data set whose labels have `n` zeros and `n` ones.
    """

    # create the data summary
    data_summary = DataSummary(data_path=data_path)

    # get the mapping from scan id to label
    label_map = data_summary._label_map

    # get the scan ids for n zeros and n ones
    zero_scan_ids = [scan_id for scan_id, label in label_map.items() if label == 0][:n]
    one_scan_ids = [scan_id for scan_id, label in label_map.items() if label == 1][:n]

    # concatenate the scan ids
    scan_ids = zero_scan_ids + one_scan_ids

    # get the labels
    labels = {scan_id: data_summary.get_label(scan_id) for scan_id in scan_ids}

    # ensure that each scan id in `zero_scan_ids` has a label of 0
    for scan_id in zero_scan_ids:
        assert (labels[scan_id] == 0), f'labels[{scan_id}] {labels[scan_id]} != 0'

    # ensure that each scan id in `one_scan_ids` has a label of 1
    for scan_id in one_scan_ids:
        assert (labels[scan_id] == 1), f'labels[{scan_id}] {labels[scan_id]} != 1'

    # create the data set
    data_set = FMRIDataset(scan_ids, labels, normalize=False, kernel_3d=None, data_path=data_path)

    # create the data loader
    data_loader = DataLoader(data_set, batch_size=16, shuffle=False)

    return data_loader