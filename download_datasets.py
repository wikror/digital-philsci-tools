import requests
import time
import os
import cProfile
import json
from urllib.parse import urlparse
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv
from tqdm import tqdm

# OUTPUT_PATH = "/home/wikror/external/semantic-scholar-corpus/corpus/"
OUTPUT_PATH = "/home/wikror/external-specter/semantic-scholar-corpus/corpus/"
DATASETS = ["embeddings-specter_v2"]#["papers", "s2orc"]#, "paper-ids", "s2orc"]

load_dotenv()
HEADERS = json.loads(os.getenv("HEADERS"))
SESSION = requests.Session() # opening session & keeping it open throughout speeds up curl requests slightly
SESSION.headers.update(HEADERS)

def download_url(url, fn, mode='wb', headers=None):
    """
    downloads a single file from a single url, in streaming mode
    url: link
    fn: final filename 
    """
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total_size = r.headers.get("content-length", 0)
        print("size to download:", total_size)
        # with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
        with open(fn + "_tmp", mode) as f:
            for chunk in r.iter_content(chunk_size=16384): 
                f.write(chunk)
                    # bar.update(16384)
        os.rename(fn + "_tmp", fn)

def download_manager(args):
    """
    for parallel processing inputs are a single iterable: (url, filename)
    downloads the file from url and writes it to file; 
    checks if file already exists & if it's incomplete tries to resume, instead of starting over 
    returns the url and performance time
    """

    t0 = time.time()
    url, fn = args[0], args[1]
    flag = False
    
    # requests.get with stream=True means that the file *should* be written in chunks, without gobbling up all the memory first
    try:
        if not os.path.exists(fn):
            if not os.path.exists(fn + "_tmp"):
                download_url(url, fn)
            else:
                resume_byte_pos = os.path.getsize(fn + "_tmp")
                download_url(url, fn, mode='ab', headers={'Range': 'bytes=%d-' % resume_byte_pos})
                
        else:
            print("File exists, skipping.")

    except Exception as e:
        flag = True
        print('Exception in download_url():', e)

    return url, time.time() - t0, flag, fn
    # the below code loads the file first into memory, then writes (we want to be memory conscious, so BAD)
    # try:
    #     r = requests.get(url)
    #     with open(fn, 'wb') as f:
    #         f.write(r.content)
    #     return(url, time.time() - t0)
    # except Exception as e:
    #     print('Exception in download_url():', e)

def download_parallel(inputs, cpus = cpu_count()):
    """
    an interface to send download_url requests in parallel, assumes cpu_count()-1 threads
    """
    bad_urls = []
    results = ThreadPool(cpus).imap_unordered(download_manager, inputs)
    for result in results:
        print('url:', result[0], 'time (s):', result[1], '\n')
        if result[2]:
            bad_urls.append((result[3]))

    return bad_urls

def main(datasets: list=DATASETS, latest: bool=True, release_id: str=None):
    """
    datasets: iterable, contains names of the datasets to download
    checks for latest release and downloads files for each dataset using parallel threads
    """
    
    if latest:
        response = SESSION.get("https://api.semanticscholar.org/datasets/v1/release")
        release_id = response.json()[-1] # we're taking the latest release

    for dataset in datasets: # we're not checking if datasets exists in given release, this can cause trouble in the future but whatever
        
        response = SESSION.get(f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/{dataset}")

        urls = response.json()["files"]
        os.makedirs(OUTPUT_PATH + release_id + "/" + dataset + "/", exist_ok=True)
        fns = [OUTPUT_PATH + release_id + "/" + dataset + "/" + os.path.basename(urlparse(u).path) for u in urls]
        
        remaining_urls = zip(urls, fns)

        while len(os.listdir(OUTPUT_PATH + release_id + "/" + dataset + "/")) != len(urls):

            while remaining_urls:
                remaining_files = download_parallel(remaining_urls, cpus=5)
                
                response = SESSION.get("https://api.semanticscholar.org/datasets/v1/release/" + release_id + "/dataset/" + dataset)
                urls = response.json()["files"]

                remaining_urls = [u for u in urls if any(f == OUTPUT_PATH + release_id + "/" + dataset + "/" + os.path.basename(urlparse(u).path) for f in remaining_files)]
            

        
if __name__ == '__main__':
    # cProfile.run('main()')
    main(latest=False, release_id="2024-08-06")