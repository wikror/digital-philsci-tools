import json
from tqdm import tqdm
import os
import gzip
from pymongo import MongoClient, InsertOne
import time

release_id = "2024-08-06"
SOURCE_PATH = f"/home/wikror/external/semantic-scholar-corpus/corpus/{release_id}/papers"
MONGO_IP = "localhost"
MONGO_PORT = 27017


def main():

    client = MongoClient(MONGO_IP, MONGO_PORT)

    if client is not None:
        # todo: here an additional (multithreading?) loop over files in dir

        db = client.papers_db
        collection = db.papers

        for filename in os.listdir(SOURCE_PATH):
            
            i = 0

            print(filename)
            requesting = []

            filepath = os.path.join(SOURCE_PATH, filename)
            with gzip.open(filepath, 'r') as file:
            

                for line in tqdm(file): # this is a jsonl file, so each line is a well-formed json containing a single paper 
                    
                    i+=1
                    
                    paper = json.loads(line)
                    # paper["_id"] = paper["corpusid"]
                    requesting.append(InsertOne(paper))

                    if i % 1000000 == 0:
                        t0 = time.time()
                        result = collection.bulk_write(requesting)
                        requesting = []
                        print("Insert time: ", time.time()-t0)

            result = collection.bulk_write(requesting)

        # create required index
        collection.create_index("corpusid")
    
        client.close()



if __name__ == '__main__':
    
    main()