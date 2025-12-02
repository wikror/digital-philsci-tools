from pymilvus import connections, MilvusClient, db, DataType
import time

MILVUS_IP = "localhost"
MILVUS_PORT = 19530

DB_NAME = "s2orcAll"
COLLECTION_NAME = "disciplineFiltered"

EMBEDDING_DIMENSION = 384 # for voyage-lite-02-instruct

DISCIPLINES = ['animalBehavior', 'neuroscience', 'psychology', 'developmentalBiology', 'ecologyEvolution', 'microbiology', 'molecularBiology', 'plantScience', 'varia']

def new_db():
    conn = connections.connect(host=MILVUS_IP, port=MILVUS_PORT)
    database = db.create_database(DB_NAME)

def new_collection(coll_name):
    # Establish connection with Milvus:
    conn = connections.connect(host=MILVUS_IP, port=MILVUS_PORT)
    # Create the database and enable it:
    db.using_database(DB_NAME)

    # Setup client for processing requests
    client = MilvusClient(
        uri='http://localhost:19530',
        token='root:Milvus',
        db_name=DB_NAME
    )

    # Set up schema for data
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
        partition_key_field="fos_id",
        num_partitions=3
    ) # primary index will be auto-generated

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="corpusid", datatype=DataType.INT64) # minimal metadata include corpusid of the paper where the sentence originates
    schema.add_field(field_name="sentence_number", datatype=DataType.INT64) # minimal metadata include the number of the sentence in the paper for easier access to context
    schema.add_field(field_name="fos_id", datatype=DataType.INT64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim = EMBEDDING_DIMENSION) # vector embedding of the sentence
    for field in DISCIPLINES:
        schema.add_field(field_name=field, datatype=DataType.BOOL)
    # schema.add_field(field_name="doc", datatype=DataType.STRING) # sentence text for ease of use
    # schema.add_field(field_name="fos", datatype=DataType.STRING) # minimal metadata include field-of-study of the paper where the sentence originates

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

    client.create_collection(
        collection_name=coll_name,
        schema = schema,
        index_params = index_params,
    )

    # wait
    time.sleep(5)

    # check status
    res = client.get_load_state(
        collection_name=coll_name
    )

    print(res)

if __name__ == '__main__':
    new_db()
    new_collection(COLLECTION_NAME)
