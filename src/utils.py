
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import time
import pprint
import preprocess

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

""" Reads in the 3D facial landmark data line by line, applying preprocessing steps outlined in research
    paper 
    Returns: np array of facial landmark data of shape (68, 3, 10k), representing 68 3D facial landmarks
    over time at a rate of 5 Hz
    np array has dimensions [[ L1x  L1y  L1z]
                             [ L2x  L2y  L2z]
                                 . . .
                             [L68x L68y L68z]]
                             where the 3rd dimension is time
"""
def process_3D_landmarks(path) -> np.ndarray:
    print(f'in utils, processing 3D facial landmark data...')
    start_time = int(time.time())
    np.set_printoptions(precision=3, suppress=True)
    max_i = 0

    ## read in landmark data
    landmarks = np.zeros((2482, 10000))
    time_idx = 0 # represents the location in the head_pose array after scaling from 30 Hz -> 5 Hz
    unsuccessful_frames = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            max_i = i
            if (i-1) % 6 != 0 or i == 0: # only include every 1 in 6 frames to reduce from 30 Hz to 5 Hz
                continue
            data = line.strip('\n').split(', ')
            if time_idx % 500 == 0:
                print(f'   processing {time_idx}th frame in the landmarks array')
            # skip any frames that were unsucessful in capturing  data
            success = int(data[3]) 
            if not success:
                unsuccessful_frames[i-1] = data
                continue
            data = [float(x) for x in data] # cvt str -> float
            data = data[4:]
            frame = np.array(data).reshape(68, 3) 
            landmarks[:, time_idx] = preprocess.process_single_landmark(frame, i-1)
            time_idx += 1
    # crop landmarks to remove zero-padding
    landmarks = landmarks[:, :time_idx]

    # helpful print statements
    # print(f'there were {len(unsuccessful_frames)} unsuccessful frames occuring at the following frames: \n{list(unsuccessful_frames.keys())}')
    # print(f'There were {len(unsuccessful_frames)} unsuccessful frames ({round(100 * len(unsuccessful_frames) / landmarks.shape[1], 3)}%) when processing the facial landmarks')
    # print(f'Landmarks was cvted from {max_i} to {time_idx} frames')
    end_time = int(time.time())
    # print(f'finished processing landmark data for a single participant in {(end_time - start_time) // 60}m {round((time.time() - start_time) % 60, 2)}s')
    
    return landmarks


""" Takes in a (68, 3) frame of landmarks, returns the frame after applying pre-processing steps 
"""
def normalize_lndmrk_frame(lndmkrks: np.ndarray) -> np.ndarray:

    pass



def process_headpose_data(path) -> np.ndarray:
    """
    Reads in head pose data line by line, applying scaling normalization (diving by 100 for Tx Ty Tz) and
    time series normalzation (going from 30 Hz (fps) to 5 Hz)
    Returns: numpy array of head pose data for the entire interview for a single participant
             numpy array has dimensions: [[Tx Ty Tx],
                                          [Rx Ry Rz]] 
              where the 3rd dimension is time
    
    Note that in error cases we skip the unsucessful frame. An alternate approach would be checking
    if neighboring frames are successful and if so, appending them
    Also note that this code processes the frames to modify indexing from 1 indexing to 0 indexing, 
    which leads to a mismatch in the source data (since the source data is never modified)
    """
    print(f'processing headpose data...')
    start_time = int(time.time())
    np.set_printoptions(precision=3, suppress=True)
    head_pose = np.zeros((2, 3, 10000))
    time_idx = 0 # represents the location in the head_pose array after scaling from 30 Hz -> 5 Hz
    unsuccessful_frames = {}

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if (i-1) % 6 != 0 or i == 0: # only include every 1 in 6 frames to reduce from 30 Hz to 5 Hz
                continue 
            data = line.strip('\n').split(', ')
            success = int(data[3]) 
            # skip any frames that were unsucessful in capturing headpose data
            if not success:
                unsuccessful_frames[i-1] = data
                continue
            data = [float(x) for x in data] # cvt str -> float
            data = data[4:]
            head_pose[:2, :3, time_idx] = np.array(data).reshape(2, 3) 
            time_idx += 1

    # Normalize Tx Ty Tz by diving by 100
    head_pose[0,:,:] /= 100
    # print(f'there were {len(unsuccessful_frames)} unsuccessful frames occuring at the following frames: \n{list(unsuccessful_frames.keys())}')
    # print(f'head_pose was cvted from {max_i} to {time_idx} frames')
    end_time = int(time.time())
    print(f'processed headpose data in {(end_time - start_time) // 60}m {round((time.time() - start_time) % 60, 2)}s')
    return head_pose

# process_headpose_data( "../data/300_P/300_CLNF_pose.txt")
process_3D_landmarks( "../data/300_P/300_CLNF_features3D.txt")