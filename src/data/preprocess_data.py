import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image as im
import torch

'''
resize_fmri: Resizes FMRI to be (140,36,64,64)
'''
# def resize_fmri(curr_fmri):
#   assert(curr_fmri.shape[0] == 140)
#   curr_reshaped_fmri = np.zeros((140, 36, 64, 64))

#   for i in range(140):
#     num_slices = curr_fmri.shape[1]
#     slice_shape = curr_fmri.shape[2:4]
#     assert(slice_shape[0] == slice_shape[1])
#     curr_mri = np.zeros((num_slices, 64, 64))
#     # resize 2d scan to 64*64
#     if slice_shape != (64,64):
#       for sl in range(num_slices):
#         curr_slice = x[i,sl]
#         curr_slice = im.fromarray(curr_slice)
#         curr_slice = curr_slice.resize((64, 64))
#         curr_slice = np.asarray(curr_slice)
#         curr_mri[sl] = curr_slice
#     else:
#       curr_mri = curr_fmri[i]

#     curr_mri_2 = np.zeros((36, 64, 64))
#     # resize number of slices to 36
#     # if num_slices != 36:
#     for sl in range(64):
#       if slice_shape[0] == 80:
#         print(f'1: {curr_fmri.shape}')
#         print(f'1.5: {curr_mri.shape}')
#         print(sl)
#       curr_slice = curr_mri[:,:,sl]
#       temp = curr_mri[:,:,sl]
#       print(f'2: {curr_slice.shape}')
#       curr_slice = im.fromarray(curr_slice)
#       print(f'3: {curr_slice.size}' )
#       print(f'3.5: {curr_slice.mode}' )
#       curr_slice = curr_slice.resize((64, 36))
#       print(f'4: {curr_slice.size}' )
#       curr_slice = np.asarray(curr_slice)
#       print(f'5: {curr_slice.shape}')
#       curr_mri_2[:,:,sl] = curr_slice
#       if (curr_reshaped_fmri.shape == curr_fmri.shape):
#         assert((temp == curr_slice).all())
#         print("HERE")
#     # else:
#     #   curr_mri_2 = curr_mri

#     curr_reshaped_fmri[i] = curr_mri_2

def create_torch(curr_fmri, fmri_folder):

  # PATH = f"./fmri-data/torch-data/data/"
  PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data/data/'


  curr_fmri_torch = torch.from_numpy(curr_fmri.astype('float64'))
  assert(curr_fmri_torch.size() == torch.Size([140,48,64,64]))
  torch.save( curr_fmri_torch,f'{PATH}/{fmri_folder}.pt')


def main():
  # path = os.getcwd() + '/fmri-data/preprocessed'
  path = '/home/ai-prac/ai-practicum/fmri-data/preprocessed'


  subject_lst = []
  image_lst = []
  # dimention_dict = {(140, 48, 80, 80):0, (140, 48, 64, 64):0, (140, 36, 64, 64):0, (140, 48, 96, 96):0}
  # d_type_counts = {'float64': 0, 'uint16': 0}
  # d_type_dims = {'float64': set(), 'uint16': set()}
  # min_max_lst = []
  subject_scans_dict = {}
  i = 1
  for subject in os.listdir(path):
    # subject_lst.append(subject)
    for fmri_folder in os.listdir(f'{path}/{subject}'):
      # image_lst.append(fmri_folder)

      # if i in [10,40]:
      curr_fmri = pickle.load( open(f'{path}/{subject}/{fmri_folder}', 'rb') )
      if curr_fmri.shape == (140, 48, 64, 64):
          create_torch(curr_fmri, fmri_folder)
          fmri_name = fmri_folder[0:7]
          print(f'{fmri_name}:  {i}  {curr_fmri.shape}')
          i += 1
          if subject in subject_scans_dict:
            subject_scans_dict[subject].append(fmri_name)
          else:
            subject_scans_dict[subject] = [fmri_name]

        # plt.pcolormesh(curr_fmri[0,0,:,:], cmap=cm.gray)
        # plt.show()
      
      
      # dimention_dict[curr_fmri.shape] += 1
      # d_type_counts[str(curr_fmri.dtype)] += 1
      # d_type_dims[str(curr_fmri.dtype)].add(curr_fmri.shape)

      # min_max_lst.append( (np.max(curr_fmri), np.min(curr_fmri) ) )


  print(subject_scans_dict)
  # PATH = f"./fmri-data/torch-data/"
  PATH = '/home/ai-prac/ai-practicum/fmri-data/torch-data/'

  pickle.dump(subject_scans_dict, open(f'{PATH}/subject-scans-dict.pickle', "wb"))


  # assert(len(subject_lst) == len(set(subject_lst)))
  # assert(len(image_lst) == len(set(image_lst)))

  # print(f'Number of Subjects: {len(subject_lst)}')
  # print(f'Number of fMRI Scans: {len(image_lst) }')

  # print(dimention_dict)
  # print(d_type_counts)
  # print(d_type_dims)
  # print(min_max_lst)


if __name__ == "__main__":
    main()
