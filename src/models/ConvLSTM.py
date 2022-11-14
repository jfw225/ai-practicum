import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pickle
import random
import itertools

def train_test_split(split, subject_scans_dict):
  assert(len(split) == 2)
  assert(sum(split) == 1)

  subject_to_num_scans = dict(zip(list(subject_scans_dict.keys()) ,map(lambda x: len(x), list(subject_scans_dict.values()))))
  # print(subject_to_num_scans)
  num_scans = sum(list(subject_to_num_scans.values()))
  # print(num_scans)
  
  all_subjects = set(list(subject_scans_dict.keys()))
  num_train_scans = 0
  train_subjects = set()
  while num_train_scans/num_scans < split[0]:
    curr_subject = random.choice( list(all_subjects - train_subjects) )
    train_subjects.add(curr_subject)
    num_train_scans += subject_to_num_scans[curr_subject]

  # print(num_train_scans/num_scans)
  test_subjects = all_subjects - train_subjects

  # assert(len(test_subjects) + len(train_subjects) == len(subject_scans_dict))

  train_scans  = []
  for subject in train_subjects:
    train_scans += subject_scans_dict[subject]

  test_scans = []
  for subject in test_subjects:
    test_scans += subject_scans_dict[subject]

  # print(len(test_scans) + len(train_scans))

  # assert(len(set(test_scans).union(set(train_scans))) == 302)

  return train_scans, test_scans



def main():
  # Create custom dataset class 


  # Create train and test split 
  PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data'
  subject_scans_dict = pickle.load( open(f'{PATH}/subject-scans-dict.pickle', 'rb') )
  # print(subject_scans_dict)

  train_data, test_data = train_test_split((.8,.2), subject_scans_dict)

  # Create labels 
  PATH = '/home/ai-prac/ai-practicum/fmri-data/'
  fmri_df = pd.read_csv(f'{PATH}/fMRI_summary.csv')
  fmri_df = fmri_df[fmri_df['Description'] == 'Resting State fMRI']
  fmri_df = fmri_df[(fmri_df['Research Group'] == 'AD') | (fmri_df['Research Group'] == 'CN')]

  keys = fmri_df["Image ID"].map(lambda x: f'I{x}')
  values = fmri_df["Research Group"]

  print(dict(zip(keys,values)))

  assert(False)

  # test_labels{}
  

  test_labels = dict()
  str_label_to_int = {'CN' :0, 'AD': 1}
  for fmri in train_data:
    str_label = fmri_df[fmri_df["Image ID"] == fmri]['Research Group']
    print(str_label)
    int_label = str_label_to_int[str_label]
    test_labels[fmri] = int_label

  train_labels = dict()
  for fmri in train_data:
    print(fmri_df[fmri_df["Image ID"] == fmri])
    str_label = fmri_df[fmri_df["Image ID"] == fmri]['Research Group']
    int_label = str_label_to_int[str_label]
    test_labels[fmri] = fmri_df[fmri_df["Image ID"] == fmri]['Research Group']

  print(len(fmri_df))




  # Create data loader

  # Implementation of model

  # Train/Evaluate functions


if __name__ == "__main__":
    main()