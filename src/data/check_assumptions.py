#!/usr/bin/env python3
import os
import pydicom

def main():
  dcm_dim_set = set()
  num_tp_set = set()
  rows_2d = set()
  cols_2d = set()
  dim_3_dcm = None
  dim_2_dcm = None
  out = None


  path = os.getcwd() + '/fmri-data/original'
  path = '/home/ai-prac/ai-practicum/fmri-data/original'
  for subject in os.listdir(path):
    # print(subject)
    for fmri_date in os.listdir(f'{path}/{subject}/Resting_State_fMRI'):
      # print(f'  {fmri_date}')
      for fmri_folder in os.listdir(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}'):
        # print(f'    {fmri_folder}')
        all_dcm = os.listdir(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}/{fmri_folder}')
        first_dcm = pydicom.dcmread(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}/{fmri_folder}/{all_dcm[0]}')

        if "I257275" == fmri_folder:
          out = first_dcm

        dcm_dim_set.add( len(first_dcm.pixel_array.shape) )

        if len(first_dcm.pixel_array.shape) == 2:
          rows_2d.add(first_dcm.Rows)
          cols_2d.add(first_dcm.Columns)
          
          # check that we know the total number of timesteps in our fMRI
          check1 = first_dcm.NumberOfTemporalPositions
          # check that we know the current timestep 
          check2 = first_dcm.TemporalPositionIdentifier
          
          check3 = -1
          # check that we know the the number of brain slices
          # Checking [Number of Slices MR] or NumberOfSlices
          for (x,y) in [(0x2001, 0x1018), (0x0054,0x0081)]:
            if [x,y] in first_dcm:
              check3 = first_dcm[x,y].value
              break
          assert(check3 != -1)

          check4 = -1
          # check that we know the the current of brain slice idx
          # Checking [Slice Number MR] or SliceVector
          for (x,y) in [ (0x2001, 0x100a) ]:
            if [x,y] in first_dcm:
              check4 = first_dcm[x,y].value
              break
          assert(check4 != -1)

          # assert (len(all_dcm) == num_slices * num_tp), f" {len(all_dcm)} != {num_slices} * {num_tp}"
          # assert (len(all_dcm) == check3 * check1), f" {len(all_dcm)} != {check3} * {check1} for {subject}, {fmri_folder}"


          dim_2_dcm = first_dcm

        elif len(first_dcm.pixel_array.shape) == 3:
          # print(len( os.listdir( f'{path}/{subject}/Resting_State_fMRI/{fmri_date}/{fmri_folder}' ) ))
          # print(len(os.listdir(f'{path}/{subject}/Resting_State_fMRI/{fmri_date}')))
          # check1 = first_dcm.NumberOfTemporalPositions
          assert(len(all_dcm) == 1)

          dim_3_dcm = first_dcm
          image_id = fmri_folder
          check1 = dim_3_dcm[0x2001, 0x1081].value




  print(out.Rows)
  print(out.Columns)
  print(out.NumberOfTemporalPositions)
  print(out.TemporalPositionIdentifier)
  
  print(out[0x2001, 0x100a].value)

  # print( dim_3_dcm[0x2001, 0x1081].value )

  print(dim_3_dcm.filename)
  # print(dim_3_dcm.Rows)
  # print(dim_3_dcm.top()[:50000])
  # print(dim_3_dcm.MREchoSequence)
#######
  # lst = list(dim_3_dcm.formatted_lines())

  # for i in lst:
  #   # if '5040' in i:
  #   #   print(i)

  #   print(i)
#######
  # # print(dim_3_dcm.TemporalPositionIndex)

  # print(dim_3_dcm.NumberOfTemporalPositions)
  # # print(dim_3_dcm.ConcatenationFrameOffsetNumber)

  # # print(dim_3_dcm.InConcatenationTotalNumber)
  # # print(dim_3_dcm.InConcatenationNumber)
  # print(dim_3_dcm.DeviceVolume)
  
  
  # print(image_id)



  # print(rows_2d)
  # print(cols_2d)

  # print(dcm_dim_set)
  

if __name__ == "__main__":
    main()
