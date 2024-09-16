# data download original python file

import os 
import time
import requests
from bs4 import BeautifulSoup
import sys
from concurrent.futures import ThreadPoolExecutor

print(f' running original file...')
start_time = time.time()

url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'
download_dir = 'data'

if not os.path.exists(download_dir):
    os.makedirs(download_dir)


# fetch webpage content
response = requests.get(url)

# print the first 100 chars to check for err code
print(f'response type: {type(response)} \nresponse: \n{response.content[0:100]}') 

soup = BeautifulSoup(response.content, 'html.parser')

links = soup.find_all('a')
print(f'{len(links)} links found. ')

data_links = [link['href'] for link in links if link['href'].endswith('.zip') or link['href'].endswith('.pdf') or link['href'].endswith('.csv')]

print(f'{len(data_links)} data links found')

print(f'ONLY DOWNLOADING THE FIRST 2 LINKS FOR DEBUGGING PURPOSES...')
data_links = data_links[:2]

print(f'example data links URLs')
[print(data_links[i]) for i in range(0, 2)]

for data_link in data_links:
# for idx in range(0, 3):
    # data_link = data_links[idx]

    if not data_link.startswith('http'):
        data_link = url + data_link
    
    file_name = os.path.join(download_dir, data_link.split('/')[-1])

    print(f'downloading {file_name} of {len(data_links)}')
    with requests.get(data_link, stream=True) as r:
        r.raise_for_status()
        with open(file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print('All files downloaded.')
end_time = time.time()
total_mins = (end_time - start_time) // 60
total_secs = (end_time - start_time) % 60
print(f'downloading {len(data_links)} took {total_mins} min, {total_secs} secs, end - start = {end_time - start_time}')


print(f'exiting now [debug]')
sys.exit()

