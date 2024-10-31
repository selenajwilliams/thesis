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

import zipfile
import shutil
def unzip_files(dir_prefix):
    failed_zips = []
    failed_moves = []
    # create a folder to move the .zip files into after extracting them

    zip_folder = f'{dir_prefix}zips'
    print(f'zip folder exists at path: {zip_folder}')
    if not os.path.exists(zip_folder):
        os.makedirs(zip_folder)
    
    i = 0
    print(f'files at this directory:')
    [print(x) for x in os.listdir(dir_prefix)]
    for zip_file in os.listdir(dir_prefix):
        if i == 500:
            break
        if 'zip' not in zip_file:
            continue
        if zip_file == "zips": # skip the zips folder when encountered
            continue
        
        # create a folder contianing the filepath if it doesn't exist
        file_name = os.path.splitext(zip_file)[0] # removes file extension (e.g. .zip, .pdf, etc)
        new_folder = f'{dir_prefix}{file_name}'

        if not os.path.exists(new_folder):
            print(f'creating folder: {new_folder}')
            os.makedirs(new_folder)
        
        current_zip_path = f'{dir_prefix}{zip_file}'
        
        # extract all zip files into the new folder
        try:
            print(f'unzipping {current_zip_path}') 
            with zipfile.ZipFile(current_zip_path, 'r') as zip_ref:
                zip_ref.extractall(new_folder)
        except Exception as e:
            print(f'error encountered when unzipping file, skipping to next file')
            failed_zips.append(zip_file)
            delete_zip_file(current_zip_path)
            continue 
        
        # move the zip file into 
        try:
            print(f'moving {zip_file} to {os.path.join(zip_folder)}')
            shutil.move(current_zip_path, os.path.join(zip_folder))
        except Exception as e:
            print(f'error encoutnered when moving file, skipping to next file')
            failed_moves.append(zip_file)
            delete_zip_file(current_zip_path)
            continue

        i += 1

    print(f'failed to unzip {len(failed_zips)} files. The failed zips include: {failed_zips}')
    print(f'failed to move {len(failed_moves)} files. The failed moves include: {failed_moves}')

    if len(failed_zips) == 0 and len(failed_moves) == 0:
        return True
    else:
        return False

def delete_zip_file(path):
    try:
        os.remove(path)  
        print(f'Deleted failed zip file: {path}')
    except Exception as delete_error:
        print(f'Error deleting file: {path}, error: {delete_error}')

    


print(f'Running download-data.py...')
download_dir = '../data'
# download_data(download_dir=download_dir) # download all data

test_dir = '../data_backup/'
res = unzip_files(test_dir) # unzip all files, deleting broken zip files (e.g. where not all chunks were downloaded)


# download_data(download_dir) # for any files that failed, 
# unzip_files(test_dir)