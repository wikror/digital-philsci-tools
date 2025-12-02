import pandas as pd
import numpy as np
import pickle, json, time, os
import itertools
# from composition_stats import ilr, closure
from skbio.stats.composition import ilr, closure
from statsmodels.stats.multivariate import test_cov
from statsmodels.multivariate.manova import MANOVA
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import PartOfSpeech

from build_bertopic_paragraph_model import *

plt.rcParams['text.usetex'] = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_PATH = f"/home/wikror/gdrive/corpus-study/data/"
query_name = "results-semantics-iteration2-2-separate-excl"
training_name = "results-semantics-iteration2-2-separate-incl"

topic_model_name = "/home/wikror/tmp/topic-model-results-semantics-iteration2-2-separate-incl-79"

JSON_PATH = f"{DATA_PATH}{query_name}.json"

DISCIPLINES = ['animalBehavior', 'neuroscience', 'psychology', 'developmentalBiology', 'ecologyEvolution', 'microbiology', 'molecularBiology', 'plantScience', 'varia']

model_name = "sentence-transformers/all-MiniLM-L6-v2"
NLP = SentenceTransformer(model_name)

# def read_results(json_path: str = JSON_PATH):
#     with open(json_path, "r") as f:
#         results = json.load(f)
#     if isinstance(results[0]["paragraph"], str):
#         for i in range(len(results)):
#             results[i]["paragraph"] = json.loads(results[i]["paragraph"])
#     return results

# def generate_embeddings(docs: list[str], embedding_model, save: bool = False, save_file: str = None):
#     """
#     Function to generate paragraph embeddings, 
#     `nlp` is the embeddings model used and can be either:
#         * `spacy.lang.en.English` for spaCy implementation, or
#         * `SentenceTransformer.SentenceTransformer` for sentence_transformers implementation
#     `docs` should be a list of strings, each string containing one paragraph in an order corresponding to results list with metadata
#     `save` is used to switch whether embeddings are saved
#     """

#     # if type(embedding_model) == SentenceTransformer:
#     embeddings = embedding_model.encode(docs, normalize_embeddings=True, show_progress_bar=True)   
#     # elif type(embedding_model) == spacy.lang.en.English:
#         # raise NotImplementedError

#     if save:
#         with open(save_file, "wb") as f:
#             pickle.dump(embeddings, f)

#     return embeddings

# def load_pretrained_embeddings(embeddings_file):
#     with open(embeddings_file, "rb") as f:
#         embeddings = pickle.load(f)
#     return embeddings

# def load_model(
#         json_path: str = JSON_PATH, 
#         embedding_model = NLP, 
#         load_dir_model: str = None,
#         load_file_embeddings: str = None,
#         ):

#     results = read_results(json_path)
#     docs = [" ".join(res["paragraph"]) for res in results]

#     embeddings = load_pretrained_embeddings(load_file_embeddings)

#     print("Loading model...")
#     st = time.time()
#     topic_model = BERTopic.load(load_dir_model)
#     print(f"Done: {time.time()-st}s")
    
#     topics, probs = topic_model.transform(documents=docs, embeddings=embeddings)

#     print("Evaluating...")
#     coherence = evaluate(docs, topic_model)

#     return docs, topics, probs, topic_model, coherence

# def evaluate(docs, topic_model, coherence_metric=["c_v", "u_mass"]):

#     cleaned_docs = topic_model._preprocess_text(docs)

#     # Extract vectorizer and analyzer from BERTopic
#     vectorizer = topic_model.vectorizer_model
#     analyzer = vectorizer.build_analyzer()

#     # Extract features for Topic Coherence evaluation
#     tokens = [analyzer(doc) for doc in cleaned_docs]
#     dictionary = corpora.Dictionary(tokens)
#     corpus = [dictionary.doc2bow(token) for token in tokens]
#     topics = topic_model.get_topics()
#     topics.pop(-1, None)
#     topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
#                 for topic in range(len(set(topics))-1)]

#     # Evaluate
#     coherence = dict()
#     for coh in coherence_metric:
#         coherence_model = CoherenceModel(topics=topic_words, 
#                                     texts=tokens, 
#                                     corpus=corpus,
#                                     dictionary=dictionary, 
#                                     coherence=coh)
#         coherence[coh] = coherence_model.get_coherence()

#     return coherence

# def generate_embeddings(docs: list[str], embedding_model, save: bool = False, save_file: str = None):
#     """
#     Function to generate paragraph embeddings, 
#     `nlp` is the embeddings model used and can be either:
#         * `spacy.lang.en.English` for spaCy implementation, or
#         * `SentenceTransformer.SentenceTransformer` for sentence_transformers implementation
#     `docs` should be a list of strings, each string containing one paragraph in an order corresponding to results list with metadata
#     `save` is used to switch whether embeddings are saved
#     """

#     # if type(embedding_model) == SentenceTransformer:
#     embeddings = embedding_model.encode(docs, normalize_embeddings=True, show_progress_bar=True)   
#     # elif type(embedding_model) == spacy.lang.en.English:
#         # raise NotImplementedError

#     if save:
#         with open(save_file, "wb") as f:
#             pickle.dump(embeddings, f)

#     return embeddings

# def topic_distribution(docs, topic_model, results, topic_nr=None):

#     topic_distr, _ = topic_model.approximate_distribution(docs)
#     topic_distr = pd.DataFrame(topic_distr, columns=[f"topic_id={i}" for i in range(topic_distr.shape[1])])

#     df = pd.DataFrame(results)

#     df = pd.concat([df, topic_distr], axis=1)

#     df.to_csv(f"{DATA_PATH}topic-dist-{query_name}-{str(topic_nr)}.csv")

#     return df


if __name__ == "__main__":
    docs, topics, probs, topic_model, coherence = load_model(json_path=f"{DATA_PATH}{training_name}.json", 
                                                             embedding_model=NLP,
                                                             load_file_embeddings=f"{DATA_PATH}embeddings-{training_name}.pickle",
                                                             load_dir_model=topic_model_name)
    
    print("Loading new results...")
    new_results = pd.DataFrame(read_results(f"{DATA_PATH}{query_name}.json"))
    new_results = new_results[~new_results.duplicated("paragraph")] # remove paragraphs that appear more than once (with multiple sentences identified in queries)
    new_docs = list(new_results["paragraph"].apply(" ".join))

    embeddings_dir = f"{DATA_PATH}embeddings-{query_name}.pickle"

    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2), token_pattern='(?u)\\b\\w\\w\\w+\\b') # by default, tokens are any 2 or more alphanumeric characters: token_pattern='(?u)\\b\\w\\w+\\b'; using token_pattern='(?u)\\b\\w\\w\\w+\\b' should limit tokens to 3+ length (ignoring most acronyms)
    pos_patterns = [
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
            [{'POS': 'NOUN'}, {'POS': 'ADJ'}],
            [{'POS': 'NOUN'}],
            [{'POS': 'ADJ'}]
    ]
    representation_model = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns)

    print("Generating new embeddings...")
    if os.path.isfile(embeddings_dir):
        new_embeddings = load_pretrained_embeddings(embeddings_dir)
    else:
        new_embeddings = generate_embeddings(docs=new_docs, embedding_model=NLP, save=True, save_file=embeddings_dir)

    print("Predicting topics...")
    topics, probs = topic_model.transform(documents=new_docs, embeddings=new_embeddings)

    print("Reducing outliers...")
    new_topics = topic_model.reduce_outliers(new_docs, topics, strategy="distributions", threshold=0.1)

    topic_model.update_topics(new_docs, topics=new_topics, vectorizer_model=vectorizer_model, representation_model=representation_model)

    print("Evaluating new data...")
    old_coherence = evaluate(docs, topic_model)
    new_coherence = evaluate(new_docs, topic_model)
    print(f"Initial coherence: {coherence}. \n Coherence for old docs: {old_coherence}. \n Coherence for new docs: {new_coherence}.")

    # get topic distribution
    print("Topic distributions...")
    topic_distr = topic_distribution(new_docs, topic_model, new_results)
    columns=[f"topic_id={i}" for i in range(len(topic_model.get_topic_info())-1)] #TODO: get the proper number of topics

    # drop outlier rows 
    topic_distr = topic_distr.drop(topic_distr[topic_distr[columns].sum(axis=1) == 0].index)

    # get topic distribution matrix as np array
    distr_matrix = topic_distr[columns].copy().to_numpy()

    # perform ilr transformation on topic distributions
    distr_matrix_ilr = closure(distr_matrix+0.000000001) # there cannot be zero values in the matrix
    distr_matrix_ilr = ilr(distr_matrix_ilr)

    # MANOVA
    topic_distr['label'] = topic_distr[DISCIPLINES].idxmax(axis=1)
    codes, levels = topic_distr["label"].factorize()
    manova = MANOVA(distr_matrix_ilr, codes)
    res = manova.mv_test()

    print(res.summary())

    eta_squared = lambda l, s: 1 - np.power(l, np.divide(1, s))

    effect_size = eta_squared(res.results["x0"]["stat"].iloc[0]["Value"], len(levels))
    print(effect_size)

    # plot topic distributions
    average_topics = topic_distr.groupby("label")[columns].mean()

    plt.figure(figsize=(40, 10))
    plt.title("Mean probability of each topic for disciplines", fontdict={"fontweight": "bold"})
    plt.plot(average_topics.T)
    locs, _ = plt.xticks()
    plt.xticks(ticks=locs, labels=range(len(columns)), rotation="vertical")
    plt.xlabel("Topic ID")
    plt.ylabel("Probability")
    plt.legend([str(i) for i in average_topics.index])
    plt.grid()
    plt.savefig(f"{DATA_PATH}top_distr-{query_name}.png", dpi=300)

    # get Cohen's d for each topic
    std_topics = topic_distr.groupby("label")[columns].std()
    counts = topic_distr.groupby("label").size()

    pairs = list(itertools.combinations(DISCIPLINES, 2))

    pooled_std = lambda n1, n2, s1, s2: np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)).reshape(-1)
    # pairs = itertools.combinations(discipline_ids.keys(), 2)
    # for pair in pairs:
    #     print(type(counts.loc[pair[0]]))
    #     print(type(counts.loc[pair[1]]))
    #     print(type(std_topics.loc[pair[0]]))
    #     print(type(std_topics.loc[pair[1]]))
    #     break
  
    pairs_std = pd.DataFrame({pair: pooled_std(counts.loc[pair[0]], counts.loc[pair[1]], std_topics.loc[pair[0]].to_numpy(), std_topics.loc[pair[1]].to_numpy()) for pair in pairs}).T

    cohen_d = lambda u1, u2, s: np.abs((u1-u2) / s)

    d_topics = pd.DataFrame({pair: cohen_d(average_topics.loc[pair[0]].to_numpy().reshape(-1), average_topics.loc[pair[1]].to_numpy().reshape(-1), pairs_std.loc[pair].to_numpy()) for pair in pairs})
    d_topics.to_csv(f"{DATA_PATH}d_topics-{query_name}.csv")

    # d_topics_ranked = pd.DataFrame()
    # for col in d_topics: 
    #     d_topics_ranked[col] = d_topics[col].nlargest(10)
    #     d_topics_ranked[str(col)+" rank"] = d_topics[col].nlargest(10).index

    # d_topics_smallest = pd.DataFrame()
    # for col in d_topics: 
    #     d_topics_smallest[col] = d_topics[col].nsmallest(10)
    #     d_topics_smallest[str(col)+" rank"] = d_topics[col].nsmallest(10).index

    # print(d_topics_ranked)
    # print(d_topics_smallest)