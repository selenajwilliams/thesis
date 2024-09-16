# a script to download the data from https://dcapswoz.ict.usc.edu/wwwdaicwoz/ 
# automating it is helpful since there are ~190 participant zip files to download

import os
import time
import requests
from bs4 import BeautifulSoup
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_zip_paths(url):
    # fetch webpage content
    response = requests.get(url)
    # print the first 100 chars to check for err code
    print(f'webpage preview: \n{response.content[0:100]}\n') 
    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all('a')
    print(f'{len(links)} links found. ')
    data_links = [link['href'] for link in links if link['href'].endswith('.zip') or link['href'].endswith('.pdf') or link['href'].endswith('.csv')]
    print(f'{len(data_links)} data links found')
    return data_links

def download_all_links(data_links, download_dir, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for data_link in data_links:
            future = executor.submit(download_link, download_dir, data_link)
            print(f'future received for data_link: {data_link}')
            futures.append(future)
        
        # wait for all tasks to complete
        for res in as_completed(futures):
            try: 
                res.result()
                print(f'res status: {data_link} {res.result()}')
            except Exception as e:
                print(f'when accessing res.result(), the following error occured: {e}')

    print(f'LOCAL: all downloads completed')



# downloads the data available at a link 
def download_link(download_dir, data_link):

    print(f'downloading data_link: {data_link} into /{download_dir} directory')

    if not data_link.startswith('http'):
        data_link = url + data_link
    
    file_name = os.path.join(download_dir, data_link.split('/')[-1])

    print(f'file created. now downloading {file_name} at url: {data_link}')
    try:
        with requests.get(data_link, stream=True, timeout=5) as r:
            r.raise_for_status()
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f'successfully wrote chunks to file: {file_name}')
        return { data_link.split('/')[-1]: "success"}
    except Exception as e:
        print(f'ERROR: Error downloading {data_link}: {e}')

        return { data_link.split('/')[-1]: f"error: {e}"}



if __name__ == "__main__":
    start_time = time.time()
    url = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/'
    download_dir = 'data'

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    data_links = fetch_zip_paths(url)

    data_links = data_links[:2] 

    print(f'example data links URLs out of {len(data_links)} data links')
    [print(data_links[i]) for i in range(0, 2)]

    print(f'downloading all links now with parallelism')
    download_all_links(data_links, download_dir, max_workers=10)    

    end_time = time.time()
    mins = (end_time - start_time) // 60
    secs = int((end_time - start_time) % 60)
    print(f'downloading {len(data_links)} took {mins} min, {secs} secs')
    print('All files downloaded.')

    # print(f'exiting now [debug]')
    # sys.exit()
