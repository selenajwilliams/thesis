
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import time
import pprint
import itertools
import re


""" This part of the file pre-processes the visual data (facial landmarks, headpose)
    Note that no pre-processing is necessary for the eye-gaze data or facial action
    units, as they are already normalized in the dataset.
"""
def process_3D_landmarks(path) -> np.ndarray:
    """ Reads in the 3D facial landmark data line by line, applying preprocessing steps outlined in research
        paper 
        Returns: np array of facial landmark data of shape (2482, num_frames), where the first dimension is 
        a vector of length 2482 containing the 68 flattened 3D facial landmarks and the average distances 
        between each pair of landmarks (which have been processed according to the paper)
        Average runtime on Mac OS (Intel Processor) is ~25-30 seconds
    """
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
                print(f'   processing {time_idx:,}th frame in the landmarks array')
            # skip any frames that were unsucessful in capturing  data
            success = int(data[3]) 
            if not success:
                unsuccessful_frames[i-1] = data
                continue
            data = [float(x) for x in data] # cvt str -> float
            data = data[4:]
            frame = np.array(data).reshape(68, 3) 
            landmarks[:, time_idx] = process_single_landmark(frame, i-1)
            time_idx += 1
    # crop landmarks to remove zero-padding
    landmarks = landmarks[:, :time_idx]

    # helpful print statements
    # print(f'there were {len(unsuccessful_frames)} unsuccessful frames occuring at the following frames: \n{list(unsuccessful_frames.keys())}')
    # print(f'There were {len(unsuccessful_frames)} unsuccessful frames ({round(100 * len(unsuccessful_frames) / landmarks.shape[1], 3)}%) when processing the facial landmarks')
    # print(f'Landmarks was cvted from {max_i} to {time_idx} frames')
    end_time = int(time.time())
    print(f'finished processing all {time_idx:,} landmark frames in {(end_time - start_time) // 60}m {round((time.time() - start_time) % 60, 2)}s')
    return landmarks

def process_single_landmark(face: np.ndarray, frame: int) -> np.ndarray:
    """
    process_single_landmark pre-pprocesses the 3D facial landmarks for a single frame of landmarks through  
    the following steps:
    1. Scale the Z-coordinate by first removing its average value (calculated over all the time steps), 
    from all the time steps
    2. Scale the coordinates so that the mean distance of each point from the origin of the coordinate 
    system is 1
    3. Compute the euclidean distance between all the possible 2278 point pairs
    4. Append this to the scaled coordinates of the facial landmarks, getting a vector of length 2482
    """
    # 1. scale Z coordinates 
    avg_z = np.mean(face[:,2]) 
    face[:,2] = face[:, 2] - avg_z 

    # 2. Normalize so mean distance from origin = 1
    x_origin, y_origin = face[33, 0], face[33, 1] # define origin as tip of nose
    xyz_orig = [x_origin, y_origin, avg_z]
    distances = np.zeros((face.shape[0], 1)) 
    # 
    for i in range(face.shape[0]):
        distances[i] = np.linalg.norm(face[i] - xyz_orig)
    avg_dist = np.mean(distances) 
    if avg_dist > 0:
        face = face / avg_dist 
        xyz_orig = xyz_orig / avg_dist
    else:
        raise ValueError(f'ERROR: at frame {frame}, the average distance between origin and other pairs is 0')

    # 3. Compute the distance between all possible pairs of landmarks 
    point_pairs = list(itertools.combinations(range(face.shape[0]), 2))
    pair_distances = np.zeros((len(point_pairs),))
    for idx, (i, j) in enumerate(point_pairs):
        pair_distances[idx] = np.linalg.norm(face[i] - face[j])

    # 4. Append the average distances among pairs to the face coordinates and return that 
    face = np.ndarray.flatten(face)
    final_vector = np.concatenate((face, pair_distances))
    return final_vector

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

def test_process_single_landmark():
    """ Tests that the per-frame processing steps function correctly
        Accuracy across time-steps was tested manually
    """
    test_avg_z()
    test_scaling_coords()
    test_distance_between_pair_points()
    test_shape_out_vec()

        # testing step 1
    def test_avg_z():
        face = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12]
            ])
        avg_z = np.mean(face[:,2])
        assert(avg_z == 7.5), 'avg z value is incorrect'
    # testing step 2
    def test_scaling_coords():
        # note that some of the correctness checks may depend on the values of test_face
        # test_face = np.array([[2, 2, 2],
        #     [2, 2, 2],
        #     [2, 2, 2],
        # ])
        # test_face = np.array([[1, 2, 3],
        #     [4, 5, 6],
        #     [7, 8, 9],
        #     [10, 11, 12],
        #     [13, 14, 15],
        #     [16, 17, 18],
        #     [19, 20, 21],
        #     [22, 23, 24]
        # ])
        test_face = np.array([[1, 1, 1], [-1, -1, -1]])


        test_origin = [0, 0, 0]
        distances = np.zeros((test_face.shape[0], 1))
        
        # calculate avg distance from origin
        for i in range(len(test_face)):
            distances[i] = np.linalg.norm(test_face[i] - test_origin)
            print(f'distance: {np.linalg.norm(test_face[i] - test_origin)}')
        avg_dist = np.mean(distances) 
        print(f'avg dist: {avg_dist}')

        # scale by avg dist to normalize so avg dist from origin = 1
        test_face = test_face / avg_dist
        test_origin = test_origin / avg_dist

        # print(f'After scaling, test face is: \n{test_face}')

        # calculate the distances from the origin again to verify correctness
        # we now expect the avg distance from the origin to be 1
        print(f'Calculating avg distance of the scaled facial values to ensure that the avg distance now = 1')
        for i in range(len(test_face)):
            distances[i] = np.linalg.norm(test_face[i] - test_origin)
            print(f'distance: {np.linalg.norm(test_face[i] - test_origin)}')
        avg_dist = np.mean(distances) 
        print(f'EXPECTING 1: avg dist: {avg_dist}')    
        # print(f'final test face: {test_face}')
    # testing step 3
    def test_distance_between_pair_points():
        # manually verified correctness
        # 3D distance calculator: https://www.calculatorsoup.com/calculators/geometry-solids/distance-two-points.php
        face = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8,9]])
        
        point_pairs = list(itertools.combinations(range(face.shape[0]), 2))
        print(f'We expect 3 pairs of landmarks (3C2); there are {len(point_pairs)}')
        pair_distances = np.zeros((len(point_pairs), 1))

        for idx, (i, j) in enumerate(point_pairs):
            pair_distances[idx] = np.linalg.norm(face[i] - face[j])
            print(f'the distance between {face[i]} and {face[j]} is {pair_distances[idx]}')
        print(f'printing computed distances...')
        [print(pair_distances[i]) for i in range(3)]
    # testing step 4
    def test_shape_out_vec():
        face = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12]])
        print(f'face.shape: {face.shape}')
        face = np.ndarray.flatten(face)
        print(f'flattened face shape: {face.shape}')
        print(f'flattened face: {face}')
        pairs = np.zeros((10,))
        final_vec = np.concatenate((face, pairs))
        print(f'final vec: {final_vec}')
        print(f'EXPECT (22,) -- final vec shape: {final_vec.shape}')

""" This part of the file pre-processes audio data
"""

""" skipping for now
"""

""" This part of the file pre-processes text data
"""

def process_transcript(path) -> list[str]:
    print(f'running process_transcript...')
    with open(path, 'r') as f:
        utterances = []
        current_start = 0
        current_end = 0
        curr_text = ""
        max_i = 0
        timer_start = time.time()

        for i, line in enumerate(f):
            max_i = i
            if i == 0:
                continue
            # if i > 150:
            #     break

            data = line.strip('\n').split('\t') # start time, stop time, text
            start_time, stop_time = float(data[0]), float(data[1])
            speaker, text = data[2].lower(), data[3].lower()
            if speaker == "participant":
                if current_start == 0:
                    current_start = start_time
                current_end = stop_time
                curr_text += f' {text}' 
            else: # Ellie speaking
                total_time = current_end - current_start
                if total_time > 1:
                    curr_text = remove_informalisms(curr_text)
                    utterances.append(curr_text)
                current_start, current_end, curr_text = 0, 0, ""
        # print(f'total text:')
        # [print(i) for i in utterances]
        
        timer_end = time.time()
        # runs in approximately 0.004s 
        print(f'processed transcript with {max_i} lines and {len(utterances)} utterances in {int((timer_end - timer_start) // 60)}m {round((timer_end - timer_start) % 60, 3)}s')
        return utterances


""" This helper function removes the informalisms from a signle line of text, by replacing contractions and
    removing 'um' and 'uh'
"""
def remove_informalisms(text: str) -> str:
    contractions_dict = {
        # slang
        "um"    : "",
        "uh"    : "",
        "y'all" : "you all",
        # contractions
        "aren't"    : "are not",
        "can't"     : "cannot",
        "couldn't"  : "could not",
        "didn't"    : "did not",
        "doesn't"   : "does not",
        "don't"     : "do not",
        "hadn't"    : "had not",
        "hasn't"    : "has not",
        "haven't"   : "have not",
        "he'd"      : "he would",  # or "he had" depending on context
        "he'll"     : "he will",
        "he's"      : "he is",     # or "he has" depending on context
        "I'd"       : "I would",   # or "I had" depending on context
        "I'll"      : "I will",
        "I'm"       : "I am",
        "I've"      : "I have",
        "isn't"     : "is not",
        "it'd"      : "it would",  # or "it had" depending on context
        "it'll"     : "it will",
        "it's"      : "it is",     # or "it has" depending on context
        "let's"     : "let us",
        "mightn't"  : "might not",
        "mustn't"   : "must not",
        "shan't"    : "shall not",
        "she'd"     : "she would", # or "she had" depending on context
        "she'll"    : "she will",
        "she's"     : "she is",    # or "she has" depending on context
        "shouldn't" : "should not",
        "that's"    : "that is",   # or "that has" depending on context
        "there's"   : "there is",  # or "there has" depending on context
        "they'd"    : "they would",# or "they had" depending on context
        "they'll"   : "they will",
        "they're"   : "they are",
        "they've"   : "they have",
        "we'd"      : "we would",  # or "we had" depending on context
        "we're"     : "we are",
        "we've"     : "we have",
        "weren't"   : "were not",
        "what'll"   : "what will",
        "what're"   : "what are",
        "what's"    : "what is",   # or "what has" depending on context
        "what've"   : "what have",
        "where's"   : "where is",  # or "where has" depending on context
        "who'd"     : "who would", # or "who had" depending on context
        "who'll"    : "who will",
        "who's"     : "who is",    # or "who has" depending on context
        "won't"     : "will not",
        "wouldn't"  : "would not",
        "you'd"     : "you would", # or "you had" depending on context
        "you'll"    : "you will",
        "you're"    : "you are",
        "you've"    : "you have",
    }
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions_dict.keys()) + r')\b')
    text = pattern.sub(lambda x: contractions_dict[x.group()], text) # substitute informalisms
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces left behind from removing 'ums'
    return text
            
    


# process_headpose_data( "../data/300_P/300_CLNF_pose.txt")
# process_3D_landmarks( "../data/300_P/300_CLNF_features3D.txt")
process_transcript("../data/300_P/300_TRANSCRIPT.csv")