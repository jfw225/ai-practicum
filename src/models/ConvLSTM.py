import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pickle
import random

def train_test_split(split, subject_scans_dict):
  assert(len(split) == 2)
  assert(sum(split) == 1)

  subject_to_num_scans = dict(zip(list(subject_scans_dict.keys()) ,map(lambda x: len(x), list(subject_scans_dict.values()))))
  print(subject_to_num_scans)
  num_scans = sum(list(subject_to_num_scans.values()))
  print(num_scans)
  
  all_train_scans = set(list(subject_scans_dict.keys()))
  num_train_scans = 0
  train_scans = set()
  while num_train_scans/num_scans < split[0]:
    curr_subject = random.choice( list(all_train_scans - train_scans) )
    train_scans.add(curr_subject)
    num_train_scans += subject_to_num_scans[curr_subject]

  print(num_train_scans/num_scans)
  test_scans = all_train_scans - train_scans

  return train_scans, test_scans



def main():
  # Create custom dataset class 


  # Create train and test split 
  PATH = '/home/ai-prac/ai-practicum/fmri-data/'
  fmri_df = pd.read_csv(f'{PATH}/fMRI_summary.csv')
  fmri_df = fmri_df[fmri_df['Description'] == 'Resting State fMRI']
  fmri_df = fmri_df[(fmri_df['Research Group'] == 'AD') | (fmri_df['Research Group'] == 'CN')]

  keys =  fmri_df["Image ID"]
  keys = keys.map(lambda x: f'I{x}')

  values = fmri_df["Research Group"]

  PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data'
  subject_scans_dict = pickle.load( open(f'{PATH}/subject-scans-dict.pickle', 'rb') )
  print(subject_scans_dict)

  train_data, test_data = train_test_split((.8,.2), subject_scans_dict)




  # Create data loader

  # Implementation of model

  # Train/Evaluate functions


if __name__ == "__main__":
    main()