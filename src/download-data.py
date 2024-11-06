# data download original python file

import os 
import time
import requests
from bs4 import BeautifulSoup
import sys
from concurrent.futures import ThreadPoolExecutor
import zipfile
import shutil

def download_data(in_path):
    start_time = time.time()
    url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'

    download_dir = f'{in_path}/zips'
    print(f'downloading DAIC-WOZ dataset to {download_dir}')

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f'created zips folder at {download_dir}')

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
            print(f'   {file_name} has already been downloaded, skipping it')
            continue

        print(f'downloading {file_name} of {len(links)}')
        # try to download the file 
        download_file(download_url, filepath, max_download_attempts=3)

    print('All files downloaded.')
    end_time = time.time()
    total_mins = (end_time - start_time) // 60
    total_secs = (end_time - start_time) % 60
    print(f'downloading {len(links)} took {total_mins} m, {round(total_secs, 4)} s')


def download_file(download_url, filepath, max_download_attempts = 3, download_attempt = 0):
        """ Helper function to download a zip file using the requests library 
            If a broken pipe error occurs, it will attempt to download the file again
            up to `max_download_attempts` times
        """
        try:
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except (requests.exceptions.RequestException, BrokenPipeError) as e:
            print(f"Failed to download {'/'.join(filepath.rsplit('/')[-2:])} initially, re-attempting for the {download_attempt +1} time")
            if os.path.exists(filepath):
                os.remove(filepath)
            while download_attempt < max_download_attempts:
                download_file(download_url, filepath, max_download_attempts, download_attempt) # could this cause an infinite recursion issue?
                download_attempt += 1
            print(f"Max retries reached. Failed to download {filepath}.")
            raise e  # Raise the error


def unzip_files(in_path, out_path):
    # dir_prefix # before, dir_prefix represnted the in path & then was modified to represent the outpath...
    failed_zips = []
    # failed_moves = []
    # create a folder to move the .zip files into after extracting them
    print(f'running unzip files to unzip all .zip files in {in_path}')

    # zip_folder = f'{dir_prefix}zips'

    # check if there are any zip files to be unzipped in this directory for error/sanity checking
    files_ending_in_zip = 0
    for file in os.listdir(in_path):
        if '.zip' in file:
            files_ending_in_zip += 1
            print(f'zip file found at file: {file}')
    if files_ending_in_zip == 0:
        print(f'No zip files found in directory {in_path}. Have you processed all of them already or listed the wrong directory path?')
        sys.exit()

    i = 0
    print(f'files at this directory:')
    [print(f'   {x}') for x in os.listdir(in_path)]
    for zip_file in os.listdir(in_path):
        if 'zip' not in zip_file:
            continue
        if zip_file == "zips": # skip the zips folder when encountered
            continue
        
        # create a folder contianing the filepath if it doesn't exist
        file_name = os.path.splitext(zip_file)[0] # removes file extension (e.g. .zip, .pdf, etc)
        participant_data_folder = f'{out_path}/{file_name}'

        if not os.path.exists(participant_data_folder):
            print(f'creating folder: {participant_data_folder}')
            os.makedirs(participant_data_folder)
        
        current_zip_path = f'{in_path}/{zip_file}'
        
        # extract all zip files into the new folder
        try:
            with zipfile.ZipFile(current_zip_path, 'r') as zip_ref:
                zip_ref.extractall(participant_data_folder)
        except Exception as e:
            # print(f'error encountered when unzipping file, skipping to next file')
            print(f'ERROR: Occurred when unzipping file {current_zip_path} \n{e}')
            failed_zips.append(zip_file)
            delete_zip_file(current_zip_path)
            continue 
        
        ## shouldn't need to move zip file any more now that we are leaving it in it's original place 
        ## and just unzipping the file
        # move the zip file into 
        # try:
        #     print(f'moving {zip_file} to {os.path.join(zip_folder)}')
        #     shutil.move(current_zip_path, os.path.join(zip_folder))
        # except Exception as e:
        #     print(f'error encoutnered when moving file, skipping to next file')
        #     failed_moves.append(zip_file)
        #     delete_zip_file(current_zip_path)
        #     continue

        i += 1

    print(f'failed to unzip {len(failed_zips)} files. The failed zips include: {failed_zips}')
    # print(f'failed to move {len(failed_moves)} files. The failed moves include: {failed_moves}')

    # if len(failed_zips) == 0 and len(failed_moves) == 0:
    #     return True
    # else:
    #     return False

def delete_zip_file(path):
    try:
        os.remove(path)  
        print(f'Deleted failed zip file: {path}')
    except Exception as delete_error:
        print(f'Error deleting file: {path}, error: {delete_error}')


print(f'Running download-data.py...')
# download_dir = '../data'
# test_dir = '../data_backup_incomplete/'

data_path = "../scratch/raw_data"
# download_data(in_path=data_path) # files will be downloaded into raw_data/zips --> download_data creates zip folder & downloads zip files there 


oscar_path = "../../scratch/raw_data"
zips_location = "../../scratch/raw_data/zips"
res = unzip_files(in_path=zips_location, out_path=oscar_path)

# download_data(download_dir) # for any files that failed, 
# unzip_files(test_dir)

# test line
# test line 2