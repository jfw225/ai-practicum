#!/usr/bin/env python3
import os
import pydicom
import pickle
import numpy as np


def parse_2d(fmri_folder_path, all_dcm, num_tp, num_slices, num_slice_key, shape):

    fmri_4d = np.zeros((num_tp, num_slices, shape[0], shape[1]))
    for dcm in all_dcm:
        curr_dcm = pydicom.dcmread(fmri_folder_path +'/'+ dcm)
        
        assert(int(curr_dcm.NumberOfTemporalPositions) == num_tp)
        assert(int(curr_dcm[num_slice_key].value) == num_slices)
        assert(int(curr_dcm.Rows) == shape[0])
        assert(int(curr_dcm.Columns) == shape[1])
        
        curr_tp = int(curr_dcm.TemporalPositionIdentifier) - 1 
        curr_slice = int(curr_dcm[0x2001, 0x100a].value) - 1
        
        fmri_4d[curr_tp, curr_slice] = curr_dcm.pixel_array
        
    return fmri_4d 

def parse_3d(fmri_folder_path, all_dcm, num_tp, shape):
    assert(len(all_dcm) == 1)

    curr_dcm = pydicom.dcmread(fmri_folder_path +'/'+ all_dcm[0])
    
    num_slices = curr_dcm.pixel_array.shape[0] / num_tp
    # print(num_tp)
    # print(curr_dcm.pixel_array.shape)
    # print(num_slices)
    # print( int(num_slices))
    assert (num_slices == int(num_slices))
    num_slices = int(num_slices)

    fmri_4d = np.zeros((num_tp, num_slices, shape[0], shape[1]))
    
    assert(int(curr_dcm.Rows) == shape[0])
    assert(int(curr_dcm.Columns) == shape[1])
    
    fmri_4d = curr_dcm.pixel_array.reshape(( num_tp, num_slices , shape[0], shape[1]))
    
    return fmri_4d


def main():
    path = os.getcwd() + '/fmri-data/original'
    shape_set = set()
    count_inc_scans = 0
    i = 1
    for subject in os.listdir(path):
        for fmri_date in os.listdir(f'{path}/{subject}/Resting_State_fMRI'):
            for fmri_folder in os.listdir(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}'):
                print(f'{i}  {fmri_folder}')
                i += 1

                # if i < 46:
                #     break

                # if fmri_folder == "I257275" or fmri_folder == "I257271":


                # 2d            3d"
                # if fmri_folder not in ["I303069", "I282008"]:
                #     break

                fmri_folder_path = f'{path}/{subject}/Resting_State_fMRI/{fmri_date}/{fmri_folder}'
                all_dcm = os.listdir(fmri_folder_path)
                # all_dcm = os.listdir(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}/{fmri_folder}')
                first_dcm = pydicom.dcmread(f'{fmri_folder_path}/{all_dcm[0]}')
                # first_dcm = pydicom.dcmread(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}/{fmri_folder}/{all_dcm[0]}')

                first_dim = len(first_dcm.pixel_array.shape)
                if first_dim == 2:

                    num_tp, num_slices, num_slice_key, curr_shape = None, None, None, None

                    num_tp = int(first_dcm.NumberOfTemporalPositions)

                    # Checking [Number of Slices MR] or NumberOfSlices
                    for key in ([0x2001, 0x1018], [0x0054,0x0081]):
                        if key in first_dcm:
                            num_slice_key = key
                            num_slices = int(first_dcm[key].value)
                            break

                    slice_key = [0x2001, 0x100a]

                    curr_shape = (first_dcm.Rows, first_dcm.Columns)

                    assert (num_slices != None)
                    assert (num_tp != None)
                    # assert (len(all_dcm) == num_slices * num_tp), f" {len(all_dcm)} != {num_slices} * {num_tp}"

                    assert(num_slice_key != None)
                    assert(curr_shape != None)

                    
                    if len(all_dcm) == num_slices * num_tp : 
                        curr_fmri_4d = parse_2d(fmri_folder_path, all_dcm, num_tp, num_slices, num_slice_key, curr_shape)
                    else:
                        count_inc_scans += 1
                        print("Reached Inc Scan")


                elif first_dim == 3:
                    num_tp, curr_shape = None, None

                    num_tp = first_dcm[0x2001, 0x1081].value
                    curr_shape = (first_dcm.Rows, first_dcm.Columns)

                    assert(num_tp != None)
                    assert(curr_shape != None)

                    curr_fmri_4d = parse_3d(fmri_folder_path, all_dcm, num_tp, curr_shape)



                else:
                    assert(False)

                # determine the path that you want to write the data to. ensure that you do
                # not:
                # - overwrite the existing data
                # - keep track of how you store the data (index it by the name or keep a 
                #   metadata file)

                shape_set.add(curr_fmri_4d.shape)

                PATH = f"./fmri-data/preprocessed/{subject}"
                
                if not os.path.exists(PATH):
                    os.mkdir(PATH)

                pickle.dump(curr_fmri_4d, open(f'{PATH}/{fmri_folder}.pickle', "wb"))


    print(shape_set)
    print(f'Number of Incomplete Scans: {count_inc_scans}')



if __name__ == "__main__":
    main()
