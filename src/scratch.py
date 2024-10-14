import time


print(f'running scratch....')
start = time.time()
time.sleep(10)

end = time.time() + 60

print(f'end: {end}, start: {start}')
mins = (end - start) // 60
secs = int((end - start) % 60)

print(f'elapsed time: {mins} m {secs} s')


test_second_frame = list(range(30))
