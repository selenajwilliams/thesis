import numpy as np
import itertools
from utils import extract_3D_landmarks

## Ocular Modalities 
# we will first do this for a single set of facial landmarks

"""
This function pre-processes the 3D facial landmarks for a single frame of landmarks through the 
following steps:
1. Scale the Z-coordinate by first removing its average value (calculated over all the time steps), 
   from all the time steps
2. Scale the coordinates so that the mean distance of each point from the origin of the coordinate 
   system is 1
3. Compute the euclidean distance between all the possible 2278 point pairs
4. Append this to the scaled coordinates of the facial landmarks, getting a vector of length 2482
"""

def process_3D_landmarks(face: np.ndarray) -> np.ndarray:
    # 1. scale Z coordinates 
    avg_z = np.mean(face[:,2]) # validated correctness with a toy array
    face[:,2] = face[:, 2] - avg_z # verified using print statements

    # 2. Normalize so mean distance from origin = 1
    x_origin, y_origin = face[33, 0], face[33, 1] # nose center
    xyz_orig = [x_origin, y_origin, avg_z]
    distances = np.zeros((face.shape[0], 1)) # from origin
    for i in range(len(face)):
        distances[i] = np.linalg.norm(face[i] - xyz_orig)
    avg_dist = np.mean(distances) 
    face = face / avg_dist 
    xyz_orig = xyz_orig / avg_dist

    # 3. Compute the distance between all possible pairs of landmarks 
    point_pairs = list(itertools.combinations(range(face.shape[0]), 2))
    pair_distances = np.zeros((len(point_pairs),))
    for idx, (i, j) in enumerate(point_pairs):
        pair_distances[idx] = np.linalg.norm(face[i] - face[j])

    # 4. Append the average distances among pairs to the face coordinates and return that 
    face = np.ndarray.flatten(face)
    final_vector = np.concatenate((face, pair_distances))
    return final_vector

def proces_headpose() -> np.ndarray:
    pass


def main():
    # data_path_3D_lndmrks = "../data/300_P/300_CLNF_features3D.txt" # local path
    data_path_3D_lndmrks = "../data/300_CLNF_features3D.txt" # OSCAR path
    face = extract_3D_landmarks(data_path_3D_lndmrks)
    face = process_3D_landmarks(face)


def test_process_3D_landmark():
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

 
       
if __name__ == "__main__":
    main()