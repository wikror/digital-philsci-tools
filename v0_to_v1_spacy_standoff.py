import json
import pickle
import requests
import gzip
import os
from tqdm import tqdm
import time
import cProfile
import multiprocessing 
from multiprocessing.pool import Pool
# import sqlite3
# from sqlite3 import Error
from pymongo import MongoClient, errors
from dotenv import load_dotenv
import spacy
from spacy.lang.en import English
import itertools
import traceback
import logging # adds up too much time, but might be useful for debugging
from sentencizer import sentencizer as sts
from langdetect import detect, LangDetectException

# GLOBAL
release_id = "2024-08-06"
OUTPUT_PATH = f"/home/wikror/external/semantic-scholar-corpus/corpus/{release_id}/s2orc-json-standoff/"
SOURCE_PATH = f"/home/wikror/external/semantic-scholar-corpus/corpus/{release_id}/s2orc/"

load_dotenv() # header value for curl requests imported from .env file
HEADERS = json.loads(os.getenv("HEADERS"))

SESSION = requests.Session() # opening session & keeping it open throughout speeds up curl requests (0.3 s compared to 0.5, which is still long)
SESSION.headers.update(HEADERS)

MONGO_IP = "localhost"
MONGO_PORT = 27017

API = False # controls if field-of-study is checked in a local database or via api call

MAX_DIRSIZE = 50000

SAMPLE_SIZE = 10000000 #for processing the full dataset setting it large so it doesn't matter

print("getting list of done files...")
DONE_FILES = {os.path.splitext(name)[0] for _dp, _dn, _names in os.walk(OUTPUT_PATH) for name in _names}
print(f"finished: {len(DONE_FILES)} papers")

NLP = English()
NLP.add_pipe("sentencizer")
NLP = sts.add_custom_sentencizer(NLP)

os.makedirs(os.path.dirname(OUTPUT_PATH+"error.log"), exist_ok=True)
logging.basicConfig(filename=OUTPUT_PATH+"error.log", level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s:%(lineno)s %(funcName)s %(message)s')

## TESTS

def test_connect_db_mongo():
    client = connect_db_mongo()
    assert isinstance(client, MongoClient), "mongo not initiated"

def test_get_metadata_local_mongo():
    client = connect_db_mongo()
    fos, metadata = get_metadata_local_mongo(client, 6389630)
    assert fos=="bio", "wrong fos"
    assert metadata["corpusid"]==6389630, "wrong metadata"

def test_write_to_file():
    output = "TEST"
    fos = "test"
    test_id = 5999222
    subdir = str(119)
    write_to_file(output, fos, test_id)
    assert os.path.isfile(OUTPUT_PATH+fos+"/"+subdir+"/"+str(test_id)+".json"), "output file not created"

# TODO:
# def test_process_dataset_file(): # check for a couple of lines if everything goes well

##

def check_fos_api(paper_id):
    """
    checks fields of study, as annotated in semantic scholar, for papers in s2orc by running curl requests to the API
    returns string, possible values: "bio" - only biology; "psych" - only psychology, "bioPsych" - biology and psychology, None - any other field of study
    """


    try:
        response = SESSION.get("https://api.semanticscholar.org/graph/v1/paper/"+paper_id+"?fields=s2FieldsOfStudy")
        fos = response.json()["s2FieldsOfStudy"]
        if any(f["category"] == "Biology" for f in fos) and any(f["category"] == "Psychology" for f in fos):
            COUNTER_BIOPSYCH+=1
            return "bioPsych"
        elif any(f["category"] == "Biology" for f in fos):
            COUNTER_BIO+=1
            return "bio"
        elif any(f["category"] == "Psychology" for f in fos):
            COUNTER_PSYCH+=1
            return "psych"
        else:
            return 0 # ignoring if not a discipline of interest
    except TypeError:
        # logging.error(f"check_fos_api Error when trying to contact the API.")
        return None

def get_metadata_local_mongo(client, paper_id):#, counter_bioPsych, counter_bio, counter_psych):
    """
    checks fields of study, as annotated in semantic scholar, for papers in s2orc using locally created MongoDB
    returns string, possible values: "bio" - only biology; "psych" - only psychology, "bioPsych" - biology and psychology, None - any other field of study
    and dict containing filtered (relevant) metadata
    """
    # print("get_metadata_local_mongo")
    keys_kept = ["corpusid", "url", "title", "authors", "year"]

    collection = client.papers_db.papers
    result = None
    found = False
    tries = 0

    while (not found) and (tries < 100):
        try:
            result = collection.find_one({"corpusid": paper_id})
            found = True
        except errors.ServerSelectionTimeoutError as e:
            print(e)
            time.sleep(1.5)
            found = False
            tries += 1
            

    if result is None:
        # logging.error(f"get_metadata_local_mongo paper {paper_id} not found in local DB")
        return None, None

    else:
        try:
            fos_list = [x["category"] for x in result["s2fieldsofstudy"]]

            if any(f == "Biology" for f in fos_list) and any(f == "Psychology" for f in fos_list):
                metadata = {key: result[key] for key in keys_kept}
                metadata["fos"] = fos_list
                metadata["journal"] = result["journal"]["name"]
                fos = "bioPsych"
                # with counter_bioPsych.get_lock():
                #     counter_bioPsych.value += 1
                
            elif any(f == "Biology" for f in fos_list):
                metadata = {key: result[key] for key in keys_kept}
                metadata["fos"] = fos_list
                metadata["journal"] = result["journal"]["name"]
                fos = "bio"
                # with counter_bio.get_lock():
                #     counter_bio.value += 1

            elif any(f == "Psychology" for f in fos_list):
                metadata = {key: result[key] for key in keys_kept}
                metadata["fos"] = fos_list
                metadata["journal"] = result["journal"]["name"]
                fos = "psych"
                # with counter_psych.get_lock():
                #     counter_psych.value += 1
            else:
                fos, metadata = None, None # ignoring if not a discipline of interest
        except TypeError:
            # logging.warning(f"get_metadata_local_mongo No FoS assigned for paper {paper_id}")
            fos, metadata = None, None
        except Exception as e:
            pass
            logging.error(f"get_metadata_local mongo: {e}", exc_info=True)

        # if fos == "bioPsych" and counter_bioPsych.value >= SAMPLE_SIZE:
        #     fos, metadata = None, None

        # if fos == "bio" and counter_bio.value >= SAMPLE_SIZE:
        #     fos, metadata = None, None
        
        # if fos == "psych" and counter_psych.value >= SAMPLE_SIZE:
        #     fos, metadata = None, None
        
        return fos, metadata

def connect_db_mongo(db_ip=MONGO_IP, db_port=MONGO_PORT):

    return MongoClient(db_ip, db_port)

def write_to_file(paper, fos, paper_id):
    """
    writing out parsed papers to corpus files
    the papers are in json, with standoff annotations for paragraphs, sections, headers, sentences
    and with metadata stored
    """
    subdir = str(int(paper_id) // MAX_DIRSIZE)
    filename = OUTPUT_PATH+fos+"/"+subdir+"/"+str(paper_id)+".json"    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(paper, f)

def preprocess_paper(paper, fos, metadata, paper_id, pdf_sha, filename, line):
    """
    parsing and extracting sentences from s2orc jsonl
    """
    # print("preprocess_paper")
    export = dict()

    export['metadata'] = metadata
    export['metadata']['fos_short'] = fos
    export["content"] = dict()
    export["content"]["text"] = paper["content"]["text"]
    export["content"]["annotations"] = paper["content"]["annotations"]
    export["content"]["annotations"]["sentences"] = []
    # parse sentences and add as standoff annotation
    tic = time.time()
    # first, segment abstract if exists

    try:
        # get abstract text
        if isinstance(export["content"]["annotations"]["abstract"], str):
            abstract_dict = json.loads(export["content"]["annotations"]["abstract"])[0]
        else:
            abstract_dict = export["content"]["annotations"]["abstract"][0]
        text =  export["content"]["text"][int(abstract_dict["start"]):int(abstract_dict["end"])]
        
        # segmenting
        first_sent_pos = int(abstract_dict["start"])
        doc = NLP(text)
        
        sent_start = 0
        first_sent = True
        
        for sent in doc.sents:
            if not first_sent: # we take the beginning of a sent as the final+1 character of the previous sent, so first and last sents need to be handled separately
                sent_end = first_sent_pos+sent[0].idx
                export["content"]["annotations"]["sentences"].append({"start": int(sent_start), "end": int(sent_end)})
            sent_start = first_sent_pos+sent[0].idx # we shift the position of the current sentence by the amount of chars corresponding to abstract start
            first_sent = False
        # last sentence
        sent_end = int(abstract_dict["end"])
        export["content"]["annotations"]["sentences"].append({"start": int(sent_start), "end": int(sent_end)})
    except TypeError as e:
        abstract_dict = None
        pass
        # logging.warning(f'preprocess_paper No abstract specified for the following paper: {filename}, line: {line}', exc_info=True)
    except Exception as e:
        pass
        logging.error(f"preprocess_paper: {e}", exc_info=True)
    
    
    
    # now segment paragraphs if exist
    try:
        if isinstance(export["content"]["annotations"]["paragraph"], str):
            paragraphs = json.loads(export["content"]["annotations"]["paragraph"])
        else:
            paragraphs = export["content"]["annotations"]["paragraph"]

        for paragraph in paragraphs:
            text = export["content"]["text"][int(paragraph["start"]):int(paragraph["end"])]
            first_sent_pos = int(paragraph["start"])

            doc = NLP(text)
            # logging.debug("segmenting sents done")

            sent_start = first_sent_pos
            first_sent = True
            
            for sent in doc.sents:
                if not first_sent: # we take the beginning of a sent as the final+1 character of the previous sent, so first and last sents need to be handled separately
                    sent_end = first_sent_pos+sent[0].idx
                    export["content"]["annotations"]["sentences"].append({"start": int(sent_start), "end": int(sent_end)})
                sent_start = first_sent_pos+sent[0].idx # we shift the position of the current sentence by the amount of chars corresponding to paragraph start
                first_sent = False
            # last sentence
            sent_end = paragraph["end"]
            export["content"]["annotations"]["sentences"].append({"start": int(sent_start), "end": int(sent_end)})

    except TypeError:
        paragraphs = None
        # logging.warning(f'preprocess_paper Incorrect formatting for the following paper: {filename}, line: {line}. Paper will be skipped.', exc_info=True)
        # print("Incorrect formatting!")
    except Exception as e:
        pass
        logging.error(f"preprocess_paper: {e}", exc_info=True)

    if paragraphs is None:
        if abstract_dict is not None:
            # logging.debug(abstract_dict["end"])
            # logging.debug(type(abstract_dict["end"]))
            text = export["content"]["text"][int(abstract_dict["end"]):]
            first_sent_pos = int(abstract_dict["end"])
        else:
            text = export["content"]["text"]
            first_sent_pos = 0 
        
        if text is not None:
            doc = NLP(text)
            # logging.debug("segmenting sents done")

            sent_start = first_sent_pos
            first_sent = True
            
            for sent in doc.sents:
                if not first_sent: # we take the beginning of a sent as the final+1 character of the previous sent, so first and last sents need to be handled separately
                    sent_end = first_sent_pos+sent[0].idx
                    export["content"]["annotations"]["sentences"].append({"start": sent_start, "end": sent_end})
                sent_start = first_sent_pos+sent[0].idx # we shift the position of the current sentence by the amount of chars corresponding to paper beginning
                first_sent = False
            # last sentence
            sent_end = len(text)
            export["content"]["annotations"]["sentences"].append({"start": sent_start, "end": sent_end})
        else:
            return None

    # TODO: segmenting if no paragraph annotation exists
    # with open("timings-custom.txt", "a") as f:
    #     f.write(str(time.time()-tic)+"\n")
    return export

def process_dataset_file(filename, connect_db=connect_db_mongo, get_metadata_local=get_metadata_local_mongo):
    """
    processes the filname file from dataset s2orc
    """
    # print("process_dataset_file")
    i = 0
    paper=""
    t0 = time.time()
    filepath = os.path.join(SOURCE_PATH, filename)
    i = 0
    with gzip.open(filepath, 'r') as file:
    # with open(filepath, 'r', encoding="utf-8") as file:
        if not API:
            conn = connect_db()
        for line in tqdm(file): # this is a jsonl file, so each line is a well-formed json containing a single paper
            paper = json.loads(line)
            fos, metadata = None, None # initializing as None in case paper_id is None
            
            if API:
                try:
                    pdf_sha = paper["content"]["source"]["pdfsha"] # we need paper_id to check field-of-study, for API it is the hash value of the source pdf
                except KeyError as e:
                    pass
                    # logging.error(f"process_dataset_file: {e} for the following contents: {paper}", exc_info=True)
                if pdf_sha is not None:
                    fos = check_fos_api(pdf_sha)   # slow, even with multithreading
            else:
                pdf_sha=None
                paper_id = paper["corpusid"]
                if paper_id is not None:
                    if paper_id in DONE_FILES:
                        continue
                    fos, metadata = get_metadata_local(conn, paper_id)#, counter_bioPsych, counter_bio, counter_psych)

            if fos is not None:
                try:
                    # See if it's non-English, and if so, skip it
                    # print(detect(paper["content"]["text"][:500]))
                    if detect(paper["content"]["text"]) != "en":
                        # if fos == "bio":
                        #     with counter_bio.get_lock():
                        #         counter_bio.value -= 1
                        # elif fos == "bioPsych":
                        #     with counter_bioPsych.get_lock():
                        #         counter_bioPsych.value -= 1
                        # elif fos == "psych":
                        #     with counter_psych.get_lock():
                        #         counter_psych.value -= 1
                        fos, metadata = None, None
                except LangDetectException as e:
                    # print(e) 
                    # langdetect choked while trying to parse this, that almost certainly
                    # means that we don't have anything here that we want
                    # if fos == "bio":
                    #     with counter_bio.get_lock():
                    #         counter_bio.value -= 1
                    # elif fos == "bioPsych":
                    #     with counter_bioPsych.get_lock():
                    #         counter_bioPsych.value -= 1
                    # elif fos == "psych":
                    #     with counter_psych.get_lock():
                    #         counter_psych.value -= 1
                    fos, metadata = None, None
                except Exception as e:
                    # if fos == "bio":
                    #     with counter_bio.get_lock():
                    #         counter_bio.value -= 1
                    # elif fos == "bioPsych":
                    #     with counter_bioPsych.get_lock():
                    #         counter_bioPsych.value -= 1
                    # elif fos == "psych":
                    #     with counter_psych.get_lock():
                    #         counter_psych.value -= 1
                    # print(e)
                    fos, metadata = None, None

            # print(fos)
            if fos is not None:
                paper = preprocess_paper(paper, fos, metadata, paper_id, pdf_sha, filename, i)
                if paper is not None:
                    write_to_file(paper, fos, paper_id)
            # else:
                # if counter_bioPsych.value >= SAMPLE_SIZE and counter_bio.value >= SAMPLE_SIZE and counter_psych.value >= SAMPLE_SIZE:
                    # break

            del paper
            i+=1

    return filename, time.time()-t0, i

def star_func(args):
    return process_dataset_file(args)

def process_dataset_parallel(cpus=multiprocessing.cpu_count()//2):
    """
    interface to process dataset files in parallel (there's about 30 large files, we're limited by CPU and IO, I think)
    """
    # print("process_dataset_parallel")
    files = os.listdir(SOURCE_PATH) # all filenames of dataset
    if os.path.exists(f"{OUTPUT_PATH}preprocessed.pickle"):
        with open(f"{OUTPUT_PATH}preprocessed.pickle", "rb") as f:
            preprocessed_files = pickle.load(f)
        files = [file for file in files if file not in preprocessed_files]
    else:
        preprocessed_files = set()

    # files.reverse()
    # print(files)
    # global counter_bioPsych
    # counter_bioPsych = multiprocessing.Value("i", 0) #for keeping track of how many are processed, only for sampling purposes
    # global counter_bio
    # counter_bio = multiprocessing.Value("i", 0)
    # global counter_psych
    # counter_psych = multiprocessing.Value("i", 0)

    results = Pool(cpus).imap_unordered(star_func, files)#zip(files, itertools.repeat(counter_bioPsych), itertools.repeat(counter_bio), itertools.repeat(counter_psych)))
    for result in results:
        # print('url:', result[0], 'time (s):', result[1], "number of lines: ", result[2])
        preprocessed_files.add(result[0])
        with open(f"{OUTPUT_PATH}preprocessed.pickle", "wb") as f:
            pickle.dump(preprocessed_files, f)
    # print(str(counter_bioPsych.value), str(counter_bio.value), str(counter_psych.value))


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # print("main")
    parallel=True
    try:
        if parallel:
            process_dataset_parallel()
        else:
            # counter_bioPsych = multiprocessing.Value("i", 0) #for keeping track of how many are processed, only for sampling purposes
            # counter_bio = multiprocessing.Value("i", 0)
            # counter_psych = multiprocessing.Value("i", 0)
            for file in os.listdir(SOURCE_PATH):
                result = process_dataset_file(file)#, counter_bioPsych, counter_bio, counter_psych)
                logging.debug(f'url: {result[0]}, time (s): {result[1]}, number of lines: {result[2]}')

    except Exception as e:
        pass
        logging.error(e, exc_info=True)
        # print(e)

if __name__ == '__main__':
    # cProfile.run('main()', "profiling-custom.prof")
    main()