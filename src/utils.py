
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

# read in the facial landmarks for the first frame
# returns a np array of 68 3D facial landmarks
def extract_3D_landmarks(path) -> np.ndarray:
    raw_data = [] # represens a single line of the file
    landmarks = np.zeros((68, 3))

    with open(path, 'r') as f:
        # if we wanted to extract multiple frames of landmarks, instead of just reading in the second line, read in the whole file or as many frames as desired
        next(f) # skips the first line
        raw_data = next(f) # read landmarks from a SINGLE (initial) frame
        if ('frame') in raw_data:
            raise Exception(f"Raw data contains file metadata; are you sure you skipped the header?")
        
        raw_data = raw_data.split(', ')
        raw_data = [float(x) for x in raw_data]
        raw_data = raw_data[4:]

        if len(raw_data) / 3 != 68:
            raise Exception(f"Data formatting error: {len(raw_data)} landmarks found; expected 68. Check data to see if it's malformatted")

        # process raw landmark CSV data into numpy array
        for i in range(0, int(len(raw_data)/3)):
            x = raw_data[i]
            y = raw_data[68 + i]
            z = raw_data[68 + 68 + i]
            landmarks[i] = [x, y, z]
    
    return landmarks


def process_headpose_data(path) -> np.ndarray:
    """
    Reads in head pose data line by line, applying scaling normalization (diving by 100 for Tx Ty Tz) and
    time series normalzation (going from 30 Hz (fps) to 5 Hz)
    Returns: numpy array of head pose data for the entire interview for a single participant
             numpy array has dimensions: [[Tx Ty Tx],
                                          [Rx Ry Rz]] 
              where the 3rd dimension is time

    """
    head_pose = np.zeros((2, 3, 10000))
    frame_idx = 0
    max_i = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            max_i = i
            if (i-1) % 6 != 0 or i == 0: # only include every 1 in 6 frames to reduce from 30 Hz to 5 Hz
                continue 
            data = line.split(', ')
            success = int(data[3]) 
            frame = int(data[0])-1  # modify to acct for 0-based indexing

            if not success:
                print(f'UNSUCCESSFUL FRAME: value for frame {frame} is 0')
                continue
            data = [float(x) for x in data] # cvt str -> float
            data = data[4:]
            # print(f'saving frame {i-1} to {i-1%6} in headpose array')
            head_pose[:2, :3, frame_idx] = np.array(data).reshape(2, 3) # frame -1 to acct for 0-based indexing
            # print(f'head_pose array at {frame_idx}: {head_pose[:,:,frame_idx]}')
            frame_idx += 1
            np.set_printoptions(precision=3, suppress=True)

    print(f'headpose[:,:,0]: {head_pose[:,:, 0]}')
    # Normalize Tx Ty Tz by diving by 100
    head_pose[0,:,:] /= 100
    print(f'headpose[:,:,0]: {head_pose[:,:, 0]}')

    # for i in range(10): # print the first 10 frames
    #     print(head_pose[:,:,i])
    print(f'head_pose was cvted from {max_i} to {frame_idx} frames')

process_headpose_data( "../data/300_P/300_CLNF_pose.txt")