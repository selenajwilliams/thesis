# data download original python file

import os 
import time
import requests
from bs4 import BeautifulSoup
import sys
from concurrent.futures import ThreadPoolExecutor

def download_data(download_dir):
    start_time = time.time()
    url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    response = requests.get(url) # fetch webpage content

    # print the first 100 chars to check for err code
    # print(f'response type: {type(response)} \nresponse: \n{response.content[0:100]}') 

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
    print(f'{len(links)} links found. ')
    links = [link['href'] for link in links if link['href'].endswith('.zip') or link['href'].endswith('.pdf') or link['href'].endswith('.csv')]
    print(f'{len(links)} links ending in .zip, .pdf, or .csv found')

    # print(f'ONLY DOWNLOADING THE FIRST 2 LINKS FOR DEBUGGING PURPOSES...')
    # links = links[:2]

    print(f'example of found items to download:')
    [print(f'   {links[i]}') for i in range(0, 2)]

    for item in links:
        # define file name, path, and download url
        if not item.startswith('http'):
            file_name = item
            download_url = url + item
        else:
            file_name = os.path.join(download_dir, item.split('/')[-1])
            download_url = item 
        filepath = f'{download_dir}/{file_name}'

        # skip download if the file has already been downloaded into the directory
        if os.path.exists(filepath):
            print(f'{file_name} has already been downloaded, skipping it')
            continue

        print(f'downloading {file_name} of {len(links)}')
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    print('All files downloaded.')
    end_time = time.time()
    total_mins = (end_time - start_time) // 60
    total_secs = (end_time - start_time) % 60
    print(f'downloading {len(links)} took {total_mins} m, {round(total_secs, 4)} s')

print(f'Running download-data.py...')
download_dir = '../data'
download_data(download_dir=download_dir)


