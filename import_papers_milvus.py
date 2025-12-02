from pymilvus import connections, MilvusClient, db, DataType
import pymilvus
import json, glob, os, itertools, pickle
import spacy, spacy_sentence_bert
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import cProfile
from sentence_transformers import SentenceTransformer
from viztracer import log_sparse
import pandas as pd
from build_milvus_db import new_collection

corpus_id = "2024-08-06"
# corpus_id = "2024-06-18"
CORPUS_PATH = f"/home/wikror/external/semantic-scholar-corpus/corpus/{corpus_id}/s2orc-json-standoff/"
DATA_PATH = f"/home/wikror/gdrive/corpus-study/data/"

MILVUS_IP = "localhost"
MILVUS_PORT = 19530

DB_NAME = "s2orcAll"
COLLECTION_NAME_PREFIX = "disciplineFiltered"

EMBEDDING_DIMENSION = 384 # for multi-qa-MiniLM-L6-cos-v1
MAX_DIRSIZE = 50000

spacy.prefer_gpu()

FOS_DICT = {"bioPsych": 0, "bio": 1, "psych": 2}

DISCIPLINES = ['animalBehavior', 'neuroscience', 'psychology', 'developmentalBiology', 'ecologyEvolution', 'microbiology', 'molecularBiology', 'plantScience', 'varia']
# DISCIPLINES = ['neuroscience']


def connect_db():
    # Establish connection with Milvus:
    conn = connections.connect(host=MILVUS_IP, port=MILVUS_PORT)
    # Enable the database:
    db.using_database(DB_NAME)

    # Setup client for processing requests
    client = MilvusClient(
        uri='http://localhost:19530',
        token='root:Milvus',
        db_name=DB_NAME
        )
    
    return client

@log_sparse
def insert_sentences(path, metadata_row, client, nlp, collection_name):
    # print(path)
    with open(path, "r") as f:
        paper = json.load(f)
    # print(paper)
    sentences = paper["content"]["annotations"]["sentences"]

    output = list()
    # print("Encoding paper...")
    all_encodings = nlp.encode([paper["content"]["text"][int(sent["start"]):int(sent["end"])] for sent in sentences], batch_size=128) # for SentenceTransformer directly
    # all_encodings = nlp.pipe([paper["content"]["text"][int(sent["start"]):int(sent["end"])] for sent in sentences]) # for spacy

    seen = set() # since some articles include duplicate sentences, we do not want to add them separately, but still want to preserve sentence absolute position for context recovery
    seen_add = seen.add # for efficiency, to minimize dynamic resolution

    for sent_num, sent in enumerate(sentences):
        sent_text = paper["content"]["text"][int(sent["start"]):int(sent["end"])]
        if not (sent_text in seen or seen_add(sent_text)):
            row = dict()
            row["corpusid"] = paper["metadata"]["corpusid"]
            row["sentence_number"] = sent_num
            row["fos_id"] = FOS_DICT[paper["metadata"]["fos_short"]]
            
            row["vector"] = all_encodings[sent_num] # for SentenceTransformer directly
            # row["vector"] = all_encodings[sent_num].vector
            for field in DISCIPLINES:
                row[field] = metadata_row[field]

            output.append(row)

    try:
        res = client.insert(
            collection_name=collection_name,
            data=output,
            # partition_name=paper["metadata"]["fos_short"]
            )
    except Exception as e:
        for i in range(0, len(output), 100):
            res = client.insert(
                collection_name=collection_name,
                data=output[i:i+100],
                # partition_name=paper["metadata"]["fos_short"]
                )

    return res
    # return None

def get_paths():
    return glob.glob(f"{CORPUS_PATH}**/*.json", recursive=True)

def star_func(args):
    return insert_sentences(*args)

def import_some():
    # import some
    for discipline in DISCIPLINES:
        if discipline == "s2orc-all":
            paths = get_paths()
        else:
            with open(DATA_PATH+discipline+"_ids.pickle", "rb") as f:
                ids = pickle.load(f)
            paths = [f"{CORPUS_PATH}{paperid[1]}/{str(int(paperid[0]) // MAX_DIRSIZE)}/{str(paperid[0])}.json" for paperid in ids.items()]
        if os.path.exists(f"{CORPUS_PATH}inserted_milvus_micro.pickle"):
            with open(f"{CORPUS_PATH}inserted_milvus_micro.pickle", "rb") as f:
                preprocessed_files = pickle.load(f)
            paths = [file for file in paths if file not in preprocessed_files]
        else:
            preprocessed_files = set()
        client = connect_db()
    
        # for SentenceTransformer directly:
        nlp = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1").cuda()
        # nlp = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

        # for spacy:
        # nlp = spacy.blank('en')
        # nlp.add_pipe('sentence_bert', config={'model_name': 'multi-qa-MiniLM-L6-cos-v1'})

        parallel = False
        if not parallel:
            i=0
            for path in tqdm(paths):
                insert_sentences(path, client, nlp, collection_name="ivf_flat")#discipline)
                preprocessed_files.add(path)
                i+=1
                # with open(f"{CORPUS_PATH}inserted_milvus_micro.pickle", "wb") as f:
                    # pickle.dump(preprocessed_files, f)
                if i >= 1000:
                    break
                    # pass
                
        else:
            cpus  = cpu_count()
            results = list(tqdm(Pool(cpus - 1).imap_unordered(star_func, zip(paths, itertools.repeat(client), itertools.repeat(nlp))), total=len(paths)))
            for result in results:
                # preprocessed_files.add(path)
                with open(f"{CORPUS_PATH}inserted.pickle", "wb") as f:
                    pickle.dump(preprocessed_files, f)
                pass

def import_all():
    
    ids = pd.read_csv(f"{DATA_PATH}fields_metadata.csv")

    paths = [f"{CORPUS_PATH}{paperid.fos}/{str(int(paperid.corpusid) // MAX_DIRSIZE)}/{str(paperid.corpusid)}.json" for _, paperid in ids.iterrows()]

    if os.path.exists(f"{CORPUS_PATH}inserted_filtered.pickle"):
        with open(f"{CORPUS_PATH}inserted_filtered.pickle", "rb") as f:
            preprocessed_files = pickle.load(f)
        paths = [file for file in paths if file not in preprocessed_files]
    else:
        preprocessed_files = set()

    client = connect_db()
    
    # for SentenceTransformer directly:
    nlp = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1").cuda()
    # nlp = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

    # for spacy:
    # nlp = spacy.blank('en')
    # nlp.add_pipe('sentence_bert', config={'model_name': 'multi-qa-MiniLM-L6-cos-v1'})

    parallel = False
    if not parallel:
        i=0 # counting articles
        coll_num = 0 # counting collections
        coll_name=COLLECTION_NAME_PREFIX+str(coll_num)
        for path in tqdm(paths):
            if i % 100000 == 0: # every 100k articles, create new collection due to memory limitations
                # save ids of articles that were inserted into previous collection
                with open(f"{CORPUS_PATH}inserted_filtered.pickle", "wb") as f:
                    pickle.dump(preprocessed_files, f)
                
                #check if a collection is loaded, and release if necessary
                if client.get_load_state(collection_name=coll_name)["state"] == pymilvus.client.types.LoadState.Loaded:
                    client.release_collection(collection_name=coll_name)
                coll_num+=1
                coll_name = COLLECTION_NAME_PREFIX+str(coll_num)
                # create new collection
                new_collection(coll_name)
                if client.get_load_state(collection_name=coll_name)["state"] != pymilvus.client.types.LoadState.Loaded: 
                    raise Exception

            if ids.iloc[i][DISCIPLINES].sum() >= 1: # limiting to only articles that are from a journal associated with one or more disciplines from the DISCIPLINES list
                insert_sentences(path, ids.iloc[i], client, nlp, collection_name=coll_name)
            preprocessed_files.add(path)
            i+=1
            
            # if i >= 1000:
                # break
                # pass
            
    else:
        raise NotImplementedError
        # cpus  = cpu_count()
        # results = list(tqdm(Pool(cpus - 1).imap_unordered(star_func, zip(paths, itertools.repeat(client), itertools.repeat(nlp))), total=len(paths)))
        # for result in results:
        #     # preprocessed_files.add(path)
        #     # with open(f"{CORPUS_PATH}inserted.pickle", "wb") as f:
        #     #     pickle.dump(preprocessed_files, f)
            # pass

def main():
    # import_some()
    import_all()

if __name__ == '__main__':
    # cProfile.run('main()', 'profiling-results-ivf_flat.pstats')
    main()
