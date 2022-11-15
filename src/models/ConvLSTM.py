import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import random
import itertools
import time

def train_test_split(split, subject_scans_dict):
  assert(len(split) == 2)
  assert(sum(split) == 1)

  subject_to_num_scans = dict(zip(list(subject_scans_dict.keys()) ,map(lambda x: len(x), list(subject_scans_dict.values()))))
  num_scans = sum(list(subject_to_num_scans.values()))
  
  all_subjects = set(list(subject_scans_dict.keys()))
  num_train_scans = 0
  train_subjects = set()
  while num_train_scans/num_scans < split[0]:
    curr_subject = random.choice( list(all_subjects - train_subjects) )
    train_subjects.add(curr_subject)
    num_train_scans += subject_to_num_scans[curr_subject]

  test_subjects = all_subjects - train_subjects

  train_scans  = []
  for subject in train_subjects:
    train_scans += subject_scans_dict[subject]

  test_scans = []
  for subject in test_subjects:
    test_scans += subject_scans_dict[subject]

  return train_scans, test_scans

class FMRIDataset(Dataset):

  def __init__(self, fmri_scan_ids ,labels):
    self.fmri_scan_ids = fmri_scan_ids
    self.labels = labels
  
  def __len__(self):
    return len(self.fmri_scan_ids)

  def __getitem__(self, fmri_scan_idx):
    fmri_scan_id = self.fmri_scan_ids[fmri_scan_idx]

    x = torch.load(f'/home/ai-prac/ai-practicum/fmri-data/torch-data/data/{fmri_scan_id}.pt')
    y = self.labels[fmri_scan_id]
    return x,y
    



def main():
  # Create custom dataset class 


  # Create train and test split 
  PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data'
  subject_scans_dict = pickle.load( open(f'{PATH}/subject-scans-dict.pickle', 'rb') )
  train_scans, test_scans = train_test_split((.8,.2), subject_scans_dict)

  # Find train and test labels 
  PATH = '/home/ai-prac/ai-practicum/fmri-data/'
  fmri_df = pd.read_csv(f'{PATH}/fMRI_summary.csv')
  fmri_df = fmri_df[fmri_df['Description'] == 'Resting State fMRI']
  fmri_df = fmri_df[(fmri_df['Research Group'] == 'AD') | (fmri_df['Research Group'] == 'CN')]
  keys = fmri_df["Image ID"].map(lambda x: f'I{x}')
  str_label_to_int = {'CN' :0, 'AD': 1}
  values = list(map(lambda x : str_label_to_int[x] , fmri_df["Research Group"]))
  all_labels = dict(zip(keys,values))
  train_labels = {key:all_labels[key] for key in train_scans}
  test_labels = {key:all_labels[key] for key in test_scans}

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
  
  print('finished training')

  # Create data loader

  # Implementation of model

  # Train/Evaluate functions


if __name__ == "__main__":
    main()