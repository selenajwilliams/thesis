import time
import numpy as np

# print(f'running scratch....')
start = time.time()
time.sleep(10)

end = time.time() + 60

print(f'end: {end}, start: {start}')
mins = (end - start) // 60
secs = int((end - start) % 60)

print(f'elapsed time: {mins} m {secs} s')

test_second_frame = list(range(30))

        

def head_pose_entire_file(path) -> np.ndarray:
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
            print(f'saving frame {i-1} to {i-1%6} in headpose array')
            head_pose[:2, :3, frame_idx] = np.array(data).reshape(2, 3) # frame -1 to acct for 0-based indexing
            # print(f'head_pose array at {frame_idx}: {head_pose[:,:,frame_idx]}')
            frame_idx += 1
            np.set_printoptions(precision=3, suppress=True)

    # for i in range(10): # print the first 10 frames
    #     print(head_pose[:,:,i])
    print(f'head_pose was cvted from {max_i} to {frame_idx} frames')

head_pose_entire_file( "../data/300_P/300_CLNF_pose.txt")