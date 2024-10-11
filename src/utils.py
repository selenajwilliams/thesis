
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

# read in the facial landmarks for the first frame
# returns a np array of 68 3D facial landmarks
def extract_landmarks(path) -> np.ndarray:
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
            raise Exception(f"Data formatting error: {num_landmarks} landmarks found; expected 68. Check data to see if it's malformatted")

        # process raw landmark CSV data into numpy array
        for i in range(0, int(len(raw_data)/3)):
            x = raw_data[i]
            y = raw_data[68 + i]
            z = raw_data[68 + 68 + i]
            landmarks[i] = [x, y, z]
    
    return landmarks