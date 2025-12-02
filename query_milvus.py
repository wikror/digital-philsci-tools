from pymilvus import connections, MilvusClient, db, DataType, utility
import pymilvus
import json, glob, os, itertools
import spacy, spacy_sentence_bert
from tqdm import tqdm
import pandas as pd
import cProfile
from sentence_transformers import SentenceTransformer
import numpy as np


corpus_id = "2024-08-06"
CORPUS_PATH = f"/home/wikror/external/semantic-scholar-corpus/corpus/{corpus_id}/s2orc-json-standoff/"
DATA_PATH = "/home/wikror/gdrive/corpus-study/data/"

MILVUS_IP = "localhost"
MILVUS_PORT = 19530

DB_NAME = "s2orcAll"
# collection_name = "molecularBiology3"
COLLECTION_NAME_PREFIX = "disciplineFiltered"
COLL_NUM = 28

EMBEDDING_DIMENSION = 384 # for multi-qa-MiniLM-L6-cos-v1

spacy.prefer_gpu()

FOS_DICT = {"bioPsych": 0, "bio": 1, "psych": 2}
FOS_DICT_INV = {v: k for k, v in FOS_DICT.items()}

DISCIPLINES = ['animalBehavior', 'neuroscience', 'psychology', 'developmentalBiology', 'ecologyEvolution', 'microbiology', 'molecularBiology', 'plantScience', 'varia']

MAX_DIRSIZE = 50000
RESULT_LIMIT = 5000

def connect_db():
    # Establish connection with Milvus:
    conn = connections.connect(host=MILVUS_IP, port=MILVUS_PORT)
    # Enable the database:
    db.using_database(DB_NAME)

    # Setup client for processing request s
    client = MilvusClient(
        uri='http://localhost:19530',
        token='root:Milvus',
        db_name=DB_NAME
        )
    return client

def query(query_doc, nlp, client, prompt=None, s_lim=10):

    print("Encoding queries...")
    query_vector = nlp.encode(query_doc, prompt=prompt)
    # query_vector = nlp()
    print("Done")

    entity_fields = ["corpusid", "sentence_number", "fos_id"]
    entity_fields.extend(DISCIPLINES)

    output = {i: list() for i in range(len(query_doc))}

    print("Querying milvus...")
    for i in range(COLL_NUM):
        print(f"Querying collection {i+1} of {COLL_NUM}...")
        collection_name = COLLECTION_NAME_PREFIX+str(i+1)
        
        if client.get_load_state(collection_name=collection_name)["state"] != pymilvus.client.types.LoadState.Loaded: # assuming no collection loaded
            client.load_collection(collection_name=collection_name)
        if client.get_load_state(collection_name=collection_name)["state"] != pymilvus.client.types.LoadState.Loaded: 
                    raise Exception

        res = client.search(
            collection_name = collection_name,
            data = query_vector,
            limit = s_lim,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=entity_fields
        )

        for j, result in enumerate(res):
            output[j].extend(result)

        client.release_collection(collection_name=collection_name)
        
        print("Done")
    print(f"Done querying. Total number of selected sentences: {sum([len(out) for out in output.values()])}.")
    # print("Number of results: ", len(output))
    return output

def get_paragraph(paper, sent_an, sent_num):

    if isinstance(paper["content"]["annotations"]["paragraph"], str):
        paragraphs = json.loads(paper["content"]["annotations"]["paragraph"])
    else:
        paragraphs = paper["content"]["annotations"]["paragraph"]

    sentences = paper["content"]["annotations"]["sentences"]

    output = []

    if paragraphs is not None:
        for paragraph in paragraphs:
            # first we need to find the paragraph that contains the sentence identified by milvus query:
            if int(paragraph["start"]) <= int(sent_an["start"]) and int(paragraph["end"]) >= int(sent_an["end"]):
                # since we don't want to sentencize again, look up all the sentences contained in the paragraph and store them as a list
                for sentence in sentences:
                    if int(paragraph["start"]) <= int(sentence["start"]) and int(paragraph["end"]) >= int(sentence["end"]):
                        output.append(paper["content"]["text"][int(sentence["start"]) : int(sentence["end"])])
                break
    if paragraphs is None or output==[]: # If there is no paragraph annotations in the doc, we just take 5 preceding and 5 following sentences (11 sentences in total)
        if sent_num-5 < 0: start = 0
        else: start = sent_num-5
        
        if sent_num+6 >= len(sentences): end = -1
        else: end = sent_num+5

        for sentence in sentences[start:end]:
            output.append(paper["content"]["text"][int(sentence["start"]) : int(sentence["end"])])

    # return the paragraph as a list of sentences:
    return output


def get_query_results(res, full_par = False, ignore_short=True):

    output = dict()
    selected_sentences = set()

    print("Getting queries texts...")
    for query_num in res.keys():
        print(f"Getting texts for query {query_num} out of {len(res.keys())}...")
        for r in tqdm(res[query_num]):
            if (r['entity']["corpusid"], r['entity']["sentence_number"]) not in output.keys():
                # print(r)
                out = dict()
                out["query"] = [query_num]
                out["distance"] = [r['distance']]
                out["corpusid"] = r['entity']["corpusid"]
                out["sentence_number"] = r['entity']["sentence_number"]
                out["fos"] = FOS_DICT_INV[r['entity']["fos_id"]]
                for discipline in DISCIPLINES:
                    out[discipline] = r['entity'][discipline]

                with open(f"{CORPUS_PATH}{out['fos']}/{str(int(out['corpusid']) // MAX_DIRSIZE)}/{str(out['corpusid'])}.json", "r") as f:
                    paper = json.load(f)

                sent_an = paper["content"]["annotations"]["sentences"][out["sentence_number"]]
                out["doc"] = paper["content"]["text"][int(sent_an["start"]):int(sent_an["end"])]

                if (not ignore_short) or (len(out["doc"]) > 16):
                    if full_par:
                        out["paragraph"] = get_paragraph(paper, sent_an, out["sentence_number"])

                    output[(r['entity']["corpusid"], r['entity']["sentence_number"])] = out
            else:
                output[(r['entity']["corpusid"], r['entity']["sentence_number"])]["query"].append(query_num)
                output[(r['entity']["corpusid"], r['entity']["sentence_number"])]["distance"].append(r['distance'])

        print(f"Currently at {len(output.values())} unique sentences.")
            
    print("Done")

    return output
        
# def write_to_json(dict, query_doc):
#     print("Saving results to file...")
#     with open(f"{DATA_PATH}results-molecular3-def-informational.json", "w") as f:
#         json.dump(dict, f, indent=4)

def top_results(res: pd.DataFrame, n: int=RESULT_LIMIT, exclusive: bool=True, queries: str="merge"):
    """
    queries: parameter which controls whether the n limit applies to
            query x discipline pairs ("separate", guarantees equal distribution of results between queries - n results for each query x discipline pair), 
            or just to disciplines ("merge", doesn't preserve the distribution of results between queries - n results for each discipline)
    exclusive: parameter to control the inclusion of papers with multiple discipline assignments
    """

    output = list()
    selected_indices = set()

    res["max_distance"] = res["distance"].apply(max)

    for discipline in DISCIPLINES:
        if exclusive:
            group_rows = res[(res[discipline]) & (res[DISCIPLINES].sum(axis=1)==1)]
        else:
            group_rows = res[res[discipline]]
        group_rows = group_rows[~group_rows.index.isin(selected_indices)]

        if queries=="merge":
            top_n_rows = group_rows.nlargest(n, "max_distance")
        if queries=="separate":
            gr_exploded = group_rows.explode(["query", "distance"])
            gr_exploded[["query","distance"]] = gr_exploded.astype({"query": "category", "distance": "float"})[["query","distance"]]
            top_n_rows = group_rows.loc[pd.concat([gr_exploded.loc[group.index].nlargest(n, "distance") for _, group in gr_exploded.groupby("query")]).index]
            top_n_rows["idx"] = top_n_rows.index
            top_n_rows = top_n_rows[~top_n_rows.duplicated("idx")].sort_values("idx")
        output.append(top_n_rows)
        selected_indices.update(top_n_rows.index)
    
    return pd.concat(output, ignore_index=True)

def desc_statistics(res):
    for discipline in DISCIPLINES:
        mean = res[res[discipline]].max_distance.mean()
        std = res[res[discipline]].max_distance.std()
        min = res[res[discipline]].max_distance.min()
        max = res[res[discipline]].max_distance.max()
        centiles = res[res[discipline]].max_distance.quantile(q=np.linspace(0.1,  0.9, 9))
        print(f"In {discipline}, mean={mean}, std={std} and the span is from {min} to {max} with centiles: \\ {centiles}")

def main():
    client = connect_db()
    
    # for SentenceTransformer directly:
    nlp = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    # for spacy:
    # nlp = spacy.blank('en')
    # nlp.add_pipe('sentence_bert', config={'model_name': 'multi-qa-MiniLM-L6-cos-v1'})

    # prompt = "Instruct: Represent the query for searching passages that contain references to the term. Query: "
    prompt = ""
    # query_doc = [
    #         "During communication, the meaning of a token is established by what, upon the token being received, enables the recipient to perform effectively a function contributing to survival, due to a mapping between the token and what it refers to.",
    #         "During communication, the meaning of a token is established by what causes the sender to produce the token in accordance with a particular response pattern, informing the recipient about the cause of the token.",
    #         "During communication, the meaning of a token is established by the structure of the signal and what it relates to, through a correlation or structural correspondence, and the goal-directed, stable, robust behavioural outcomes it produces and has produced in the past.",
    #         "During communication, the meaning of a token is established by what it carries mutual information about, allowing the recipient in a systematic way to act appropriately, and improving the performance.",
    #         "During communication, the meaning of a token is established by the information that is causally necessary for the recipient to maintain its own existence over time.",
    #         "During communication, the meaning of a token is established by how it changes the perceived probabilities for the recipient.",
    #         "During communication, the meaning of a token is established by the structural correspondence between the token and its target, in such a way that the correspondence is causally relevant to the success of a recipient's strategy, and can fail and be corrected.",
    #         ]

    df = pd.read_csv(f"{DATA_PATH}mean_queries.csv", sep=";")
    query_doc = df["doc"].tolist()
    
    res = query(query_doc[:int(np.floor(len(query_doc)/6))], nlp, client, s_lim=16384, prompt=prompt)
    output = pd.DataFrame.from_dict(get_query_results(res, full_par=True), orient="index")
    del res
    
    print("Results inclusive...")
    incl_output = top_results(output, exclusive=False, queries="merge")
    desc_statistics(incl_output)
    print("Saving results to file...")
    incl_output.to_json(f"{DATA_PATH}results-semantics-iteration2-2-examples-1-merge-incl.json", orient='records')
    del incl_output

    print("Results exclusive...")
    excl_output = top_results(output, exclusive=True, queries="merge")
    del output
    desc_statistics(excl_output)
    print("Saving results to file...")
    excl_output.to_json(f"{DATA_PATH}results-semantics-iteration2-2-examples-1-merge-excl.json", orient='records')
    # output.to_csv(f"{DATA_PATH}results-signaling-adaptationist-inclusive.csv")
    # write_to_json(output, query)

if __name__ == '__main__':
    # cProfile.run('main()', 'profiling-results')
    main()