
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

def extract_headpose(path) -> np.ndarray:
    """
    frame, timestamp, confidence, success, Tx, Ty, Tz, Rx, Ry, Rz
    1, 0, 0.939744, 1, 69.3181, 39.2286, 575.033, 0.203683, -0.0885823, -0.0504782
    """

    head_pose = np.zeros((2, 3))

    with open(path, 'r') as f:
        next(f)
        raw_data = next(f)
        if ('frame') in raw_data:
            raise Exception(f"Raw data contains file metadata; are you sure you skipped the header?")

        raw_data = raw_data.split(', ')
        print(f'raw data: {raw_data}')
        raw_data = [float(x) for x in raw_data]
        if raw_data[3] != 0: # check if success metadata != 0 
            raw_data = raw_data[4:] # remove frame, timestamp, confidence, success fields
            head_pose[:2, :3] = np.array(raw_data).reshape(2, 3)
        else:
            print(f'success metric = 0, should skip to next line here')

        print(f'head pose: \n{head_pose}')

    return head_pose