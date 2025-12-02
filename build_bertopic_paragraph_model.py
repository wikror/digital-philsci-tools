import json, pickle
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, PartOfSpeech, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os, time
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from gensim.models.coherencemodel import CoherenceModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

spacy.prefer_gpu()

DATA_PATH = f"/home/wikror/gdrive/corpus-study/data/"

query_name = "results-semantics-iteration2-2-separate-incl"
JSON_PATH = f"{DATA_PATH}{query_name}.json"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_name = "intfloat/e5-large-v2"

# Paths:
EMBEDDINGS_PATH = f"{DATA_PATH}embeddings-{query_name}-sent-par.pickle"

# figure params & defaults
plt.rcParams['text.usetex'] = True

pio.renderers.default = "png+browser"

fig = go.Figure(layout={'title': 'Figure Title',
                        'font': {'family': 'Noto Sans'}})
fig.update_layout(
    font_family="Noto Sans",
    font_color="black",
    title_font_family="Noto Sans",
    title_font_color="black",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)
templated_fig = pio.to_templated(fig)
pio.templates['sans_figure'] = templated_fig.layout.template
pio.templates.default = 'sans_figure'

# for spacy:z
# spacy.prefer_gpu()
# NLP = spacy.blank('en')
# NLP.add_pipe('sentence_bert', config={'model_name': model_name})

# for sentence_transformers:
NLP = SentenceTransformer(model_name)

def read_results(json_path: str = JSON_PATH):
    with open(json_path, "r") as f:
        results = json.load(f)
    if isinstance(results[0]["paragraph"], str):
        for i in range(len(results)):
            results[i]["paragraph"] = json.loads(results[i]["paragraph"])
    return results

def generate_embeddings(docs: list[str], embedding_model, save: bool = False, save_file: str = None):
    """
    Function to generate paragraph embeddings, 
    `nlp` is the embeddings model used and can be either:
        * `spacy.lang.en.English` for spaCy implementation, or
        * `SentenceTransformer.SentenceTransformer` for sentence_transformers implementation
    `docs` should be a list of strings, each string containing one paragraph in an order corresponding to results list with metadata
    `save` is used to switch whether embeddings are saved
    """

    # if type(embedding_model) == SentenceTransformer:
    embeddings = embedding_model.encode(docs, normalize_embeddings=True, show_progress_bar=True)   
    # elif type(embedding_model) == spacy.lang.en.English:
        # raise NotImplementedError

    if save:
        with open(save_file, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings

def load_pretrained_embeddings(embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings
    
def build_model(
        json_path: str = JSON_PATH, 
        embedding_model = NLP, 
        save_model: bool = False,
        # save_file_model: str = None,
        save_embeddings: bool = False, 
        save_file_embeddings: str = None, 
        load_embeddings: bool = False, 
        load_file_embeddings: str = None,
        nr_topics = None,
        calculate_probabilities: bool = False,
        ):

    print("Reading results...")
    results = read_results(json_path)
    # docs = [" ".join(res["paragraph"]) for res in results]
    # docs = [res["doc"] for res in results]
    docs = [sent for res in results for sent in res["paragraph"]]
    print("Embeddings...")
    if load_embeddings:
        embeddings = load_pretrained_embeddings(load_file_embeddings)
    else:
        embeddings = generate_embeddings(docs=docs, embedding_model=embedding_model, save=save_embeddings, save_file=save_file_embeddings)

    # we can specify some of the parameters for dimensionality reduction and clustering:
    dimensionality_reducer = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', low_memory=False, random_state=42) # defaults: UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', low_memory=False, random_state=42)
    # dimensionality_reducer = PCA(n_components=5, random_state=42) # defaults: PCA(n_components=5, random_state=42)
    cluster_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom', prediction_data=True) # defaults: HDBSCAN(min_cluster_size=150, metric='euclidean', cluster_selection_method='eom', prediction_data=True); for "communication" I used min_cluster_size=50
    # cluster_model = KMeans(n_clusters=nr_topics, random_state=42)
    # representation_model = KeyBERTInspired()
    pos_patterns = [
            [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
            [{'POS': 'NOUN'}, {'POS': 'ADJ'}],
            [{'POS': 'NOUN'}],
            [{'POS': 'ADJ'}]
    ]

    pos_patterns2 = [
            [{'POS': 'VERB'}],
            [{'POS': 'ADV'}, {'POS': 'VERB'}],
            [{'POS': 'VERB'}, {'POS': 'ADV'}]
    ]

    main_representation_model = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns, top_n_words=30)
    alternative_representation_model1 = [KeyBERTInspired(top_n_words=50), MaximalMarginalRelevance(diversity=.5, top_n_words=30)]
    alternative_representation_model2 = PartOfSpeech("en_core_web_sm", pos_patterns=pos_patterns2, top_n_words=30)

    representation_model = {
        "Representation": main_representation_model,
        "Nouns and modifiers": main_representation_model,
        "KeyBERT + MMR": alternative_representation_model1,
        "Verbs and modifiers": alternative_representation_model2,
        }

    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2), token_pattern='(?u)\\b\\w\\w\\w+\\b') # by default, tokens are any 2 or more alphanumeric characters: token_pattern='(?u)\\b\\w\\w+\\b'; using token_pattern='(?u)\\b\\w\\w\\w+\\b' should limit tokens to 3+ length (ignoring most acronyms)

    topic_model = BERTopic(
                        umap_model=dimensionality_reducer, 
                        hdbscan_model=cluster_model, 
                        representation_model=representation_model, 
                        vectorizer_model=vectorizer_model,
                        nr_topics=nr_topics,
                        calculate_probabilities=calculate_probabilities,
                        embedding_model=embedding_model,
                        top_n_words=10,
                        )
    
    print("Fitting model...")
    topic_model = topic_model.fit(documents=docs, embeddings=embeddings)

    topics, probs = topic_model.transform(documents=docs, embeddings=embeddings)

    # print(topic_model.get_topic_info())

    print("Reducing outliers...")
    # outlier reduction:
    # new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
    # new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings, threshold=0.1)
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="distributions", threshold=0.1)
    # new_topics = topic_model.reduce_outliers(docs, topics, strategy="probabilities", probabilities=probs, threshold=0.05)

    topic_model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer_model)#, representation_model=representation_model)

    if save_model:
        print("Saving model...")
        embedding_model = model_name
        topic_model.save(f"{DATA_PATH}topic_model-{query_name}-sent-par-{str(nr_topics)}.safetensors", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

    print(topic_model.get_topic_info())
    
    print("Evaluating...")
    coherence = evaluate(docs, topic_model)

    return docs, topics, probs, topic_model, coherence

def load_model(
        json_path: str = JSON_PATH, 
        embedding_model = NLP, 
        load_dir_model: str = None,
        load_file_embeddings: str = None,
        ):
    results = read_results(json_path)
    # docs = [" ".join(res["paragraph"]) for res in results]
    # docs = [res["doc"] for res in results]
    docs = [sent for res in results for sent in res["paragraph"]]

    embeddings = load_pretrained_embeddings(load_file_embeddings)
    print("Loading model...")
    st = time.time()
    topic_model = BERTopic.load(load_dir_model)
    print(f"Done: {time.time()-st}s")
    topics, probs = topic_model.transform(documents=docs, embeddings=embeddings)
    print("Evaluating...")
    coherence = evaluate(docs, topic_model)
    return docs, topics, probs, topic_model, coherence

def visualize(topic_model, docs, embeddings, fig_dir = DATA_PATH, topic_nr=None):
    # fig = topic_model.visualize_topics()
    # fig.write_html(fig_dir+f"/vis_topics-{query_name}-{str(topic_nr)}.html")
    # fig.write_image(fig_dir+f"/vis_topics-{query_name}-{str(topic_nr)}.png", scale=3)

    fig = topic_model.visualize_documents(docs, embeddings=embeddings)
    fig.write_html(fig_dir+f"/vis_documents-{query_name}-sent-par-{str(topic_nr)}.html")
    fig.write_image(fig_dir+f"/vis_documents-{query_name}-sent-par-{str(topic_nr)}.png", scale=3)

    hierarchical_topics = topic_model.hierarchical_topics(docs)
    fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig.write_html(fig_dir+f"/vis_hierarchy-{query_name}-sent-par-{str(topic_nr)}.html")
    fig.write_image(fig_dir+f"/vis_hierarchy{query_name}-sent-par-{str(topic_nr)}.png", scale=3)

    fig = topic_model.visualize_barchart(top_n_topics=20, n_words=7)
    fig.write_html(fig_dir+f"/vis_barchart-{query_name}-sent-par-{str(topic_nr)}.html")
    fig.write_image(fig_dir+f"/vis_barchart-{query_name}-sent-par-{str(topic_nr)}.png", scale=3)

    if len(topic_model.get_topic_info()) > 8:
        fig = topic_model.visualize_heatmap(n_clusters=8)
        fig.write_html(fig_dir+f"/vis_heatmap-{query_name}-sent-par-{str(topic_nr)}.html")
        fig.write_image(fig_dir+f"/vis_heatmap-{query_name}-sent-par-{str(topic_nr)}.png", scale=3)

    # fig =  topic_model.visualize_probabilities()
    # fig.write_html(fig_dir+f"/vis_probs-{query_name}.html")

    # topic_distr, topic_token_distr = topic_model.approximate_distribution(
    #     docs, calculate_tokens=True
    #     )
    
    # df = topic_model.visualize_approximate_distribution(docs[0], topic_token_distr[0])
    # with open(fig_dir+f"/vis_approx-dist-{query_name}.html", "w") as f:
    #     f.write(df.to_html())

def topic_distribution(docs, topic_model, results, topic_nr=None):

    topic_distr, _ = topic_model.approximate_distribution(docs)
    topic_distr = pd.DataFrame(topic_distr, columns=[f"topic_id={i}" for i in range(topic_distr.shape[1])])

    df = pd.DataFrame(results)

    df = pd.concat([df, topic_distr], axis=1)

    df.to_csv(f"{DATA_PATH}topic-dist-{query_name}-sent-par-{str(topic_nr)}.csv")

    return df

def evaluate(docs, topic_model, coherence_metric=["c_v", "u_mass"]):

    cleaned_docs = topic_model._preprocess_text(docs)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = topic_model.get_topics()
    topics.pop(-1, None)
    topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                for topic in range(len(set(topics))-1)]

    # Evaluate
    coherence = dict()
    for coh in coherence_metric:
        coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence=coh)
        coherence[coh] = coherence_model.get_coherence()

    return coherence

def test_topic_number(range_min=5, range_max=205, step=25):
    coherences = dict()
    for i in range(range_min, range_max, step):
        start_time = time.time()
        if os.path.exists(EMBEDDINGS_PATH):
            docs, topics, probs, topic_model, coherence = build_model(load_embeddings=True, load_file_embeddings=EMBEDDINGS_PATH, nr_topics=i, calculate_probabilities=False)
        else:
            docs, topics, probs, topic_model, coherence = build_model(save_embeddings=True, save_file_embeddings=EMBEDDINGS_PATH, nr_topics=i, calculate_probabilities=False)
        coherences[i] = coherence
        print(f"Coherence for {i} topics is: {coherence}. Elapsed: {time.time()-start_time}s.")

    # for i in range(50, 171, 10):
    #     docs, topics, probs, topic_model, coherence = build_model(load_embeddings=True, load_file_embeddings=EMBEDDINGS_PATH, nr_topics=i)
    #     coherences[i] = coherence
    #     print(f"Coherence for {i} topics is: {coherence}")

    # for i in range(175, 250, 25):
    #     docs, topics, probs, topic_model, coherence = build_model(load_embeddings=True, load_file_embeddings=EMBEDDINGS_PATH, nr_topics=i)
    #     coherences[i] = coherence
    #     print(f"Coherence for {i} topics is: {coherence}")

    pd.DataFrame.from_dict(coherences, orient="index").to_csv(f"{DATA_PATH}cohs-{query_name}-{range_min}-{range_max}-{step}.csv")

def main(topic_nr = None, load_embeddings = None):
    """
    Function for constructing a new topic model. 
    Can use previously generated embeddings and accepts a specification of the number of topics after reduction
    """
    if topic_nr is None:
        if not load_embeddings:
            docs, topics, probs, topic_model, coherence = build_model(save_embeddings=True, 
                                                                      save_file_embeddings=EMBEDDINGS_PATH, 
                                                                      save_model=True, 
                                                                    #   save_file_model=f"{DATA_PATH}model-{query_name}.safetensors"
                                                                      )
        else:
            docs, topics, probs, topic_model, coherence = build_model(load_embeddings=True, 
                                                                      load_file_embeddings=EMBEDDINGS_PATH, 
                                                                      save_model=True, 
                                                                    #   save_file_model=f"{DATA_PATH}model-{query_name}.safetensors"
                                                                      )
    else:
        if not load_embeddings:
            docs, topics, probs, topic_model, coherence = build_model(save_embeddings=True, 
                                                                      save_file_embeddings=EMBEDDINGS_PATH, 
                                                                      save_model=True, 
                                                                    #   save_file_model=f"{DATA_PATH}model-{query_name}.safetensors", 
                                                                      nr_topics=topic_nr
                                                                      )
        else:
            docs, topics, probs, topic_model, coherence = build_model(load_embeddings=True, 
                                                                      load_file_embeddings=EMBEDDINGS_PATH, 
                                                                      save_model=True, 
                                                                    #   save_file_model=f"{DATA_PATH}model-{query_name}.safetensors", 
                                                                      nr_topics=topic_nr
                                                                      )

    # model_details(docs, topic_model)

    return docs, topics, probs, topic_model, coherence

def pretrained(model_file: str):
    """
    Function for loading a pretrained topic model.
    """
    docs, topics, probs, topic_model, coherence = load_model(load_file_embeddings=EMBEDDINGS_PATH, 
                                                             load_dir_model=model_file)
    
    print(f"Coherence for the loaded model is: {coherence}.")
    model_details(docs, topic_model)

    return docs, topics, probs, topic_model, coherence

def model_details(docs, topic_model, topic_nr="auto"):
    """
    Function to generate visualizations and document-level topic distributions.
    """
    topic_model.get_topic_info().to_csv(f"{DATA_PATH}topic-info-{query_name}-sent-par-{topic_nr}.csv")

    embeddings = load_pretrained_embeddings(EMBEDDINGS_PATH)
    results = read_results(JSON_PATH)

    print ("Distributions...")
    topic_distribution(docs, topic_model, results, topic_nr)

    print("Visualizations...")
    visualize(topic_model = topic_model, docs = docs, embeddings = embeddings, topic_nr=topic_nr)

    topic_model.get_document_info(docs, df=pd.DataFrame(results).explode("paragraph").drop("paragraph", axis=1)).to_csv(f"{DATA_PATH}document-info-{query_name}-sent-par-{topic_nr}.csv")


    
if __name__ == "__main__":

    docs, topics, probs, topic_model, coherence = main()
    # docs, topics, probs, topic_model, coherence = main(topic_nr=5, load_embeddings=True)
    # model_details(docs, topic_model, 5)
    # docs, topics, probs, topic_model, coherence = main(topic_nr=79, load_embeddings=True)
    # model_details(docs, topic_model, 79)
    # docs, topics, probs, topic_model, coherence = main(load_embeddings=True)
    print(topic_model.get_topic_info())
    print(evaluate(docs,topic_model))
    # model_details(docs, topic_model)
    
    # nr_topics = 5
    # docs, topics, probs, topic_model, coherence = pretrained(f"{DATA_PATH}topic_model-{query_name}-{nr_topics}.safetensors")
    # nr_topics = 50
    # docs, topics, probs, topic_model, coherence = pretrained(f"{DATA_PATH}topic_model-{query_name}-{nr_topics}.safetensors")
    # nr_topics = str(None)
    # docs, topics, probs, topic_model, coherence = pretrained(f"{DATA_PATH}topic_model-{query_name}-{nr_topics}.safetensors")

    test_topic_number(5, 226, 10)

    # test_topic_number(121, 124, 1)
    
    # print(f"Topic coherence: {coherence}")
    # print(topic_model.get_topic_info())