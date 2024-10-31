
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
import time
import pprint
import itertools
import re
import os
# imports for covarep processing
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing import sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # this removes warnings about rebuilding tensorflow with compiler flags. If TF is being too slow, comment this out and follow instructions from warning to improve runtime


""" This part of the file pre-processes the visual data (facial landmarks, headpose)
    Note that no pre-processing is necessary for the eye-gaze data or facial action
    units, as they are already normalized in the dataset.
"""
def process_3D_landmarks(in_dir, out_dir, ID) -> np.ndarray:
    """ Reads in the 3D facial landmark data line by line, applying preprocessing steps outlined in research
        paper 
        Returns: np array of facial landmark data of shape (2482, num_frames), where the first dimension is 
        a vector of length 2482 containing the 68 flattened 3D facial landmarks and the average distances 
        between each pair of landmarks (which have been processed according to the paper)
        Average runtime on Mac OS (Intel Processor) is ~25-30 seconds
    """

    in_path = f'{in_dir}{ID}_P/{ID}_CLNF_features3D.txt' # e.g. ../data/raw_data/300_P/300_CLNF_features3D.txt
    out_path = f'{out_dir}/3DLandmarks/{ID}.npy' # e.g. ../data/processed_data/train/3DLandmarks/300.npy

    np.set_printoptions(precision=3, suppress=True)

    landmarks = np.zeros((2482, 10000))
    time_idx = 0 # represents the location in the head_pose array after scaling from 30 Hz -> 5 Hz
    unsuccessful_frames = {}
    with open(in_path, 'r') as f:
        for i, line in enumerate(f):
            max_i = i
            if (i-1) % 6 != 0 or i == 0: # only include every 1 in 6 frames to reduce from 30 Hz to 5 Hz
                continue
            data = line.strip('\n').split(', ')
            if time_idx % 1000 == 0:
                print(f'   processing {time_idx:,}th frame in the landmarks array')

            success = int(data[3]) 
            if not success:
                unsuccessful_frames[i-1] = data
                continue
            data = [float(x) for x in data] # cvt str -> float
            data = data[4:]
            frame = np.array(data).reshape(68, 3) 
            landmarks[:, time_idx] = process_single_landmark(frame, i-1)
            time_idx += 1
    landmarks = landmarks[:, :time_idx] # crop landmarks to remove zero-padding

    print(f'saving 3D facial landmarks with padding ({round(landmarks.nbytes / (1024 **2), 3)} Mb) for participant {ID} to {out_path}')
    # if not os.path.exists(f'{out_dir}/3DLandmarks'):
    #     os.mkdir(f'{out_dir}/3DLandmarks')
    np.save(out_path, landmarks) 

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

def process_headpose_data(DIR, ID) -> np.ndarray:
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
    path = f'{DIR}{ID}_P/{ID}_CLNF_pose.txt'
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
    # print(f'head_pose was cvted from {5} to {time_idx} frames')
    end_time = int(time.time())
    # print(f'processed headpose data in {(end_time - start_time) // 60}m {round((time.time() - start_time) % 60, 2)}s')
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

def process_covarep_data(DIR, ID) -> np.ndarray:
    """ This functionality is adapted from Arbaaz Qureshi, author of Gender-Aware 
        Estimation of Depression Severity Level in a Multimodal Setting, from
        this github repo: 
        https://github.com/arbaazQureshi/DAIC_WOZ_data_preprocessing/blob/master/acoustic/COVAREP/training_data/preprocessing.py
    """
    covarep_path = f'{DIR}{ID}_P/{ID}_COVAREP.csv'
    transcript_path = f'{DIR}{ID}_P/{ID}_TRANSCRIPT.csv'

    def preprocess():
        max_frames = -1
        min_frames = 1000000000
        
        print(ID, end='\r')
        # for now we are hard coding the paths, but this can be made modular to handle all paths by doing: 
        # data/'+str(ID)+'_P/'+str(ID)+'_COVAREP.csv' 
        # data/'+str(ID)+'_P/'+str(ID)+'_TRANSCRIPT.csv' 
        data = pd.read_csv(covarep_path, header=None)
        transcript = pd.read_csv(transcript_path, sep='\t')

        data = data.values
        transcript = transcript.values
                
        transcript = transcript[transcript[:,2] == 'Participant']
        transcript = transcript[:, [0,1]]
        transcript = (transcript*100 + 0.5).astype(int)

        participant_speech_features = []
        
        for i in range(transcript.shape[0]):
            
            start_range = transcript[i,0]-15
            end_range = transcript[i,1]+15
                
            #if(end_range - start_range + 1 > 300):
            participant_speech_features = participant_speech_features + data[start_range: end_range+1].tolist()
        
        participant_speech_features = np.array(participant_speech_features)
        participant_speech_features = participant_speech_features[participant_speech_features[:,1] == 1]

        participant_speech_features[:, 0:1] = preprocessing.scale(participant_speech_features[:, 0:1])
        participant_speech_features[:, 2:] = preprocessing.scale(participant_speech_features[:, 2:])

        participant_speech_features = np.hstack((participant_speech_features[:, 0:1], participant_speech_features[:, 2:]))

        a = np.arange(participant_speech_features.shape[0])
        participant_speech_features = participant_speech_features[a%4 == 0]

        no_of_frames = participant_speech_features.shape[0]

        if(max_frames < no_of_frames):
            max_frames = no_of_frames

        if(min_frames > no_of_frames):
            min_frames = no_of_frames

        # save each processed accoustic modality to a binary file if desired
        # np.save(f'{outpath}/{ID}_COVAREP.npy', participant_speech_features)
        return participant_speech_features 
    
    def pad(processed_unpadded_covarep_features):
        X = []
        X.append(sequence.pad_sequences(processed_unpadded_covarep_features, maxlen=22000, dtype='float32', padding='pre').T.tolist())

        X = np.array(X)
        X = np.squeeze(X, axis=0)
        return X

    covarep_features = preprocess() # pre process
    covarep_features = pad(covarep_features) # add padding

    return covarep_features


def process_formant_data(DIR, ID) -> np.ndarray:
    """ Processes formant data for a single participant interview 
        Credit to: Arbaaz Qureshi, original implementation available at:
        https://github.com/arbaazQureshi/DAIC_WOZ_data_preprocessing/tree/master
    """
    path = f'{DIR}{ID}_P/{ID}_FORMANT.csv'


    def preprocess():
        max_frames = -1
        min_frames = 1000000000
        # modify to go through each participant
        data = pd.read_csv(path, header=None)

        data = data.values
        a = np.arange(data.shape[0])

        data = preprocessing.scale(data)
        data = data[a%10 == 0]

        if(max_frames < data.shape[0]):
            max_frames = data.shape[0]

        if(min_frames > data.shape[0]):
            min_frames = data.shape[0]

        return data    

    
    def pad(formant_data):
            X = []
            X.append(sequence.pad_sequences(formant_data, maxlen=20000, dtype='float32', padding='pre').T.tolist())
            X = np.array(X)
            X = np.squeeze(X, axis=0)
            return X
    
    formant_data = preprocess()
    formant_data = pad(formant_data)
    # np.save(outpath, formant_data) # optionally save the padded data to a file (this will redundantly save a lot of 0s to files -- consider modifying)
    return formant_data


""" This part of the file pre-processes text data
"""

""" process_transcript processes the text data through the following steps:
    1. Parse the transcript for participant speech, saving all utterances >1 second
       Note: We define an utterance as any block of user speech totaling > 1 second in time;
             multiple consecutive participant lines in the transcript form a single utterance
    2. Replace informalisms in user speech with proper english (e.g. ain't becomes are not)
    3. Pass each sentence into a universal sentence encoder, resulting in a 512 dimensional 
       vector for each utterance
       Note: We consider each utterance to be a sentence 
       
    Returns: np array with dimensions (512, 4,000) where each row is a 512-dimension embedding 
             of a sentence. We pad along the time-axis with zeros to achieve uniform shape of 
             (512, 4000)
"""
def process_transcript(DIR, ID) -> np.ndarray:
    path = f'{DIR}{ID}_P/{ID}_TRANSCRIPT.csv'

    embeddings = np.zeros((512, 400))
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

        embeddings[:,:len(utterances)] = get_sentence_embedding(utterances)
        
        timer_end = time.time()
        # runs in approximately 0.004s 
        # print(f'processed transcript with {max_i} lines and {len(utterances)} utterances into an {embeddings.shape} embeddings array in {int((timer_end - timer_start) // 60)}m {round((timer_end - timer_start) % 60, 3)}s')
        return embeddings

""" Given a list of utterances, returns an embedding of all the utterances
    returns a (512, len(utterances)) np array where 512 is the shape of
    the embedding of each utterance, and len(utterances) means that there
    is a row/embedding for every utternace
    Adapted from this tensorflow tutorial: https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder
"""
def get_sentence_embedding(utterances: list[str]) -> np.ndarray:
    from absl import logging
    import tensorflow_hub as hub
    logging.set_verbosity(logging.ERROR)
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"    
    # print(f'loading model now...') # uncomment to see how long it takes to load the model (~a couple seconds)
    model = hub.load(module_url)
    embedding = model(utterances) # run inference on the text, which expects a list of msgs

    # uncomment to see the message, embedding shape (512), and embedding snippet for each utterance
    # for i, message_embedding in enumerate(np.array(embedding).tolist()):
    #     print("Message: {}".format(utterances[i]))
    #     print("Embedding size: {}".format(len(message_embedding)))
    #     message_embedding_snippet = ", ".join(
    #         (str(x) for x in message_embedding[:3]))
    #     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

    embedding = embedding.numpy() # cvt to np array and reshape for dimensions (512,)
    embedding = np.squeeze(embedding.T)
    return embedding

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



""" This part of the file gets the participant IDs and handles controller logic like preprocessing each of the files
"""
def get_train_dev_test_IDs():
    """ This removes incomplete data from the train, dev, test split
        IDs are used in pre-processing to systematically access the raw data paths based on participant ID
        The DAIC-WOZ documentation indices that the IDs in `incomplete_data_ID_list` are faulty and contain 
        incomplete data, so we omit them from the dataset we will use to train the model
    """
    train_set_ID_list = [303, 304, 305, 310, 312, 313, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 330, 333, 336, 338, 339, 340, 341, 343, 344, 345, 347, 348, 350, 351, 352, 353, 355, 356, 357, 358, 360, 362, 363, 364, 366, 368, 369, 370, 371, 372, 374, 375, 376, 379, 380, 383, 385, 386, 391, 392, 393, 397, 400, 401, 409, 412, 414, 415, 416, 419, 423, 425, 426, 427, 428, 429, 430, 433, 434, 437, 441, 443, 445, 446, 447, 448, 449, 454, 455, 456, 457, 459, 463, 464, 468, 471, 473, 474, 475, 478, 479, 485, 486, 487, 488, 491]
    dev_set_ID_list = [302, 307, 331, 335, 346, 367, 377, 381, 382, 388, 389, 390, 395, 403, 404, 406, 413, 417, 418, 420, 422, 436, 439, 440, 451, 458, 472, 476, 477, 482, 483, 484, 489, 490, 492]
    test_set_ID_list = [300, 301, 306, 308, 309, 311, 314, 323, 329, 332, 334, 337, 349, 354, 359, 361, 365, 378, 384, 387, 396, 399, 405, 407, 408, 410, 411, 421, 424, 431, 432, 435, 438, 442, 450, 452, 453, 461, 462, 465, 466, 467, 469, 470, 481]
    incomplete_data_ID_list = [342, 394, 398, 460, 373, 444, 451, 458, 480, 402]

    def prune(IDs_list, faulty_IDs_list):
        for ID in faulty_IDs_list:
            if ID in IDs_list:
                IDs_list.remove(ID)
        return IDs_list
    
    train_set_ID_list = prune(train_set_ID_list, incomplete_data_ID_list)
    dev_set_ID_list = prune(dev_set_ID_list,     incomplete_data_ID_list)
    test_set_ID_list = prune(test_set_ID_list,   incomplete_data_ID_list)
    return train_set_ID_list, dev_set_ID_list, test_set_ID_list

def process_data(in_dir, out_dir, set_type, IDs_list):
    """ Processes data for all participants for a single set of data (e.g. for the train, test, or dev set)
    """
    IDs_list = [300] # testing with a single participant with working data # TODO: remove this

    for idx, ID in enumerate(IDs_list):
        print(f'processing ID {ID} in {set_type} data out of {len(IDs_list)} {set_type} IDs')
        out_dir = f'{out_dir}/{set_type}' # e.g. processed_data/train or processed_data/test
        landmarks = process_3D_landmarks(in_dir, out_dir, ID)
        # head_pose_data = process_headpose_data(in_dir, ID)
        # covarep_data = process_covarep_data(in_dir, ID)
        # formant_data = process_formant_data(in_dir, ID)
        # transcript_data = process_transcript(in_dir, ID)

        # total_size = (landmarks.nbytes + head_pose_data.nbytes + covarep_data.nbytes + formant_data.nbytes) / (1024 **2 )
        # print(f'successfully pre processed all modalities for participant with ID {ID}, containing {round(total_size, 4)} MB')


def init_directories():
    """ For easy set up / system portability, we initialize the processed data directories where the data pre-processing 
        functions will save data binaries of the processed, padded data. 
        In each folder (e.g. ../data/processed_data/train/3DLandmarks), there will be a list of files titled ID.npy 
        which represents the processed + padded data for the user with that ID
    """
    prefix = '../data/processed_data'
    paths = [
        prefix,
        f'{prefix}/train/3DLandmarks',
        f'{prefix}/train/headpose'
        f'{prefix}/train/covarep'
        f'{prefix}/train/formant'
        f'{prefix}/train/transcript'
        f'{prefix}/dev/3DLandmarks',
        f'{prefix}/dev/headpose'
        f'{prefix}/dev/covarep'
        f'{prefix}/dev/formant'
        f'{prefix}/dev/transcript'
        f'{prefix}/test/3DLandmarks',
        f'{prefix}/test/headpose'
        f'{prefix}/test/covarep'
        f'{prefix}/test/formant'
        f'{prefix}/test/transcript'
    ]

    for path in paths:
        if not os.path.exists(path):
            os.makdir(path)
        else:
            print(f'in initializing directories, {path} already exists')

def main():
    """ Process data, save the output to files
        Each function requires 2 inputs: 
        `dir` the directory prefix where participant data is stored (e.g. '../data/)
        `ID` the ID of the particpant that is currently being processed
    """

    train_set_IDs, dev_set_IDs, test_set_IDs = get_train_dev_test_IDs()
    print(f'processing training data...')
    process_data(in_dir='../data/raw_data/', out_dir='../data/processed_data', set_type='train', IDs_list=train_set_IDs)
    print(f'finished processing training data')

    # process_data(in_dir='../data/raw_data/', out_dir='../data/processed_data', set_type='dev', IDs_list=dev_set_IDs)
    # print(f'finished processing development/validation data')

    # process training data


    # process development (validation) data
    

    # process test data



if __name__ == "__main__":
    print(f'running main() function')
    main()


# process_headpose_data( "../data/300_P/300_CLNF_pose.txt")
# process_3D_landmarks( "../data/300_P/300_CLNF_features3D.txt")
# process_transcript("../data/300_P/300_TRANSCRIPT.csv")
# process_covarep_data('', '')
# process_formant_data('', '')