import time
import numpy as np
import os

# def init_directories():
#     """ For easy set up / system portability, we initialize the processed data directories where the data pre-processing 
#         functions will save data binaries of the processed, padded data. 
#         In each folder (e.g. ../data/processed_data/train/3DLandmarks), there will be a list of files titled ID.npy 
#         which represents the processed + padded data for the user with that ID
#     """
#     prefix = '../data/processed_data'
#     paths = [
#         '../data',
#         prefix,
#         f'{prefix}/train',
#         f'{prefix}/dev',
#         f'{prefix}/test',
#         f'{prefix}/train/3DLandmarks',
#         f'{prefix}/train/headpose',
#         f'{prefix}/train/covarep',
#         f'{prefix}/train/formant',
#         f'{prefix}/train/transcript',
#         f'{prefix}/dev/3DLandmarks',
#         f'{prefix}/dev/headpose',
#         f'{prefix}/dev/covarep',
#         f'{prefix}/dev/formant',
#         f'{prefix}/dev/transcript',
#         f'{prefix}/test/3DLandmarks',
#         f'{prefix}/test/headpose',
#         f'{prefix}/test/covarep',
#         f'{prefix}/test/formant',
#         f'{prefix}/test/transcript',
#     ]

#     for path in paths:
#         if not os.path.exists(path):
#             os.mkdir(path)
#         else:
#             print(f'{path} already exists')

# init_directories()

# """ Counts how many files & folders are in a given folder/path
#     Used to check if my OSCAR filebase is the same as what I have locally
# """
# def count_items_in_folder(path):
#     len_items = len(os.listdir(path))
#     print(f'there are {len_items} in {path}')
#     return len_items

# path = "../data/raw_data/raw_data"
# count_items_in_folder(path)

string = "../data/raw_data/raw_data/formant_features.csv"
res = string.rsplit("/")[1]
print(f'{res}')