import time
from pymilvus import connections, MilvusClient, db, DataType, FieldSchema, CollectionSchema
import pymilvus
import json, os, pickle
from tqdm import tqdm
import pandas as pd

# Configuration
corpus_id = "2024-08-06"
EMBEDDINGS_PATH = f"/home/wikror/external-specter/semantic-scholar-corpus/corpus/{corpus_id}/embeddings-specter_v2/"
DATA_PATH = f"/home/wikror/gdrive/corpus-study/data/"

# Database connection
MILVUS_IP = "localhost"
MILVUS_PORT = 19530

DB_NAME = "s2orcFullPaperEmbeddings"
COLLECTION_NAME = "paperEmbeddings"

EMBEDDING_DIMENSION = 768  # Semantic Scholar specter embeddings are 768-dim
BATCH_SIZE = 1000

def ensure_database():
    """Check if database exists and create it if necessary"""
    try:
        # Connect to Milvus server
        connections.connect(host=MILVUS_IP, port=MILVUS_PORT)
        
        # Check if database exists by trying to use it
        try:
            db.using_database(DB_NAME)
            print(f"Database '{DB_NAME}' already exists")
        except Exception:
            # Database doesn't exist, create it
            print(f"Creating database '{DB_NAME}'...")
            db.create_database(DB_NAME)
            db.using_database(DB_NAME)
            print(f"Database '{DB_NAME}' created successfully")
            
    except Exception as e:
        raise RuntimeError(f"Failed to ensure database exists: {e}")

def connect_db():
    """Connect to Milvus database"""
    connections.connect(host=MILVUS_IP, port=MILVUS_PORT)
    db.using_database(DB_NAME)

    client = MilvusClient(
        uri=f'http://{MILVUS_IP}:{MILVUS_PORT}',
        token='root:Milvus',
        db_name=DB_NAME
    )
    
    return client

def create_collection(client):
    """Create Milvus collection for paper embeddings or resume import if it exists"""
    # Check if collection already exists
    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists - resuming import for remaining files")
        
        # Load the collection to ensure it's available
        client.load_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' loaded successfully")
        
        return True  # Return True to indicate resuming
    
    # Collection doesn't exist, create it
    print(f"Creating new collection: {COLLECTION_NAME}")
    
    # Define minimal schema - just corpusid and vector
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="corpusid", dtype=DataType.INT64),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
    ]
    
    schema = CollectionSchema(fields, description="Paper embeddings from Semantic Scholar")

    # Set up indexing
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="corpusid",
        index_type="INVERTED"
    )

    index_params.add_index(
        field_name="vector", 
        index_type="IVF_PQ",
        params={"nlist":1024,"m":4,"nbits":8},
        metric_type="COSINE",
    )
    
    # Create new collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params = index_params,
    )

    # wait
    time.sleep(5)

    # check status
    res = client.get_load_state(
        collection_name=COLLECTION_NAME
    )
    
    if res.get("state") == "Loaded":
        print(f"Created collection: {COLLECTION_NAME}")
        return False  # Return False to indicate new collection
    else:
        raise RuntimeError(f"Failed to create collection '{COLLECTION_NAME}': {res}")

def get_embedding_files():
    """Get all embedding files from Semantic Scholar dataset"""
    # All files in embeddings_path are relevant, get all files regardless of extension
    files = []
    for root, dirs, filenames in os.walk(EMBEDDINGS_PATH):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} embedding files")
    return files

def get_s2orc_corpus_ids():
    """Get set of corpus IDs that exist in S2ORC from MongoDB or cached pickle file"""
    pickle_file = DATA_PATH + "s2orc_corpus_ids.pickle"
    
    # Check if cached file exists
    if os.path.exists(pickle_file):
        print("Loading S2ORC corpus IDs from cached file...")
        with open(pickle_file, "rb") as f:
            s2orc_ids = pickle.load(f)
        print(f"Loaded {len(s2orc_ids)} S2ORC corpus IDs from cache")
        return s2orc_ids
    
    # Cache doesn't exist, query MongoDB
    from pymongo import MongoClient
    
    print("Loading S2ORC corpus IDs from MongoDB...")
    
    mongo_client = MongoClient("localhost", 27017)
    db = mongo_client["papers_db"]
    collection = db["papers"]
    
    # Get all corpus IDs where s2orc is True - no need to specify corpusid field
    s2orc_ids = set()
    cursor = collection.find({"s2orc": True}, {"corpusid": 1})
    
    for doc in cursor:
        if "corpusid" in doc:
            s2orc_ids.add(doc["corpusid"])
    
    mongo_client.close()
    print(f"Loaded {len(s2orc_ids)} S2ORC corpus IDs from MongoDB")

    # Save to cache for future use
    print("Saving S2ORC corpus IDs to cache...")
    with open(pickle_file, "wb") as f:
        pickle.dump(s2orc_ids, f)
    print("Cache saved successfully")

    return s2orc_ids

def process_embedding_file(filepath, client, s2orc_ids):
    """Process a single embedding file and insert only S2ORC papers into Milvus"""
    import gzip
    
    # Determine if file is gzipped
    is_gzipped = filepath.endswith('.gz')
    
    batch_data = []
    processed_count = 0
    skipped_count = 0
    
    try:
        # Open file (gzipped or regular)
        if is_gzipped:
            file_handle = gzip.open(filepath, 'rt', encoding='utf-8')
        else:
            file_handle = open(filepath, 'r', encoding='utf-8')
            
        with file_handle as f:
            for line in f:
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    
                    # Extract corpusid, model, and vector according to actual format
                    corpusid = data.get('corpusid')
                    vector_str = data.get('vector')  # vector is stored as string
                    
                    if corpusid is None or vector_str is None:
                        continue
                    
                    # Parse vector string to get actual list of floats
                    vector = json.loads(vector_str)
                    
                    # CHECK: Only process if paper is in S2ORC
                    if int(corpusid) not in s2orc_ids:
                        skipped_count += 1
                        continue
                        
                    # Prepare row for insertion
                    row = {
                        "corpusid": int(corpusid),
                        "vector": vector
                    }
                    
                    batch_data.append(row)
                    processed_count += 1
                    
                    # Insert batch when it reaches BATCH_SIZE
                    if len(batch_data) >= BATCH_SIZE:
                        client.insert(
                            collection_name=COLLECTION_NAME,
                            data=batch_data
                        )
                        batch_data = []
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line in {filepath}: {e}")
                    continue
                    
        # Insert remaining data
        if batch_data:
            client.insert(
                collection_name=COLLECTION_NAME,
                data=batch_data
            )
            
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return 0, 0
        
    return processed_count, skipped_count

def import_embeddings():
    """Main import function"""
    print("Starting paper embedding import from Semantic Scholar dataset...")
    
    # Ensure database exists
    ensure_database()
    
    # Connect to database
    client = connect_db()
    
    # Create collection or resume import
    is_resuming = create_collection(client)
    
    # Load progress tracking
    progress_file = f"{EMBEDDINGS_PATH}imported_files.pickle"
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            processed_files = pickle.load(f)
        if is_resuming:
            print(f"Resuming import: {len(processed_files)} files already processed")
        else:
            print(f"Warning: Found existing progress file but collection was new. {len(processed_files)} files marked as processed")
    else:
        processed_files = set()
        if is_resuming:
            print("Warning: Collection exists but no progress file found. Starting fresh import tracking.")
    
    # Load S2ORC paper IDs
    print("Loading S2ORC paper IDs from MongoDB...")
    s2orc_ids = get_s2orc_corpus_ids()
    print(f"Loaded {len(s2orc_ids)} S2ORC paper IDs")
    
    # Get embedding files
    embedding_files = get_embedding_files()
    unprocessed_files = [f for f in embedding_files if f not in processed_files]
    
    print(f"Processing {len(unprocessed_files)} files...")
    
    total_processed = 0
    total_skipped = 0
    
    # Process each file
    for filepath in tqdm(unprocessed_files, desc="Processing files"):
        try:
            processed, skipped = process_embedding_file(filepath, client, s2orc_ids)
            total_processed += processed
            total_skipped += skipped
            processed_files.add(filepath)
            
            print(f"Processed {os.path.basename(filepath)}: {processed} processed, {skipped} skipped")
            
            # Save progress periodically
            if len(processed_files) % 10 == 0:
                with open(progress_file, "wb") as f:
                    pickle.dump(processed_files, f)
                    
        except Exception as e:
            print(f"Failed to process {filepath}: {e}")
            continue
    
    # Final progress save
    with open(progress_file, "wb") as f:
        pickle.dump(processed_files, f)
    
    print("\nImport completed successfully!")
    print(f"Total embeddings processed: {total_processed}")
    print(f"Total embeddings skipped (not in S2ORC): {total_skipped}")
    print(f"Files processed: {len(processed_files)}")
    
    
    # Load collection for searching
    try:
        client.load_collection(COLLECTION_NAME)
        print("✅ Collection loaded and ready for search!")
    except Exception as e:
        print(f"⚠️ Failed to load collection: {e}")

def main():
    """Main execution function"""
    import_embeddings()

if __name__ == '__main__':
    from datetime import datetime
    main()
    print(f"\n{'='*60}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")