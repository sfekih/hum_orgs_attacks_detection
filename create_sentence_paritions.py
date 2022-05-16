from typing import List
import numpy as np
import pandas as pd

import re
import json

import nltk

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer

porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words())

en_icrc_stop_words = ["icrc", "red", "cross", "crescent", "committee", "redcross"]
for word in en_icrc_stop_words:
    stop_words.add(word)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation

import umap.umap_ as umap
import hdbscan

import networkx as nx
import community.community_louvain as community
from ast import literal_eval
import argparse
from tqdm import tqdm
import operator
import warnings

warnings.filterwarnings(action="ignore")


parser = argparse.ArgumentParser()

parser.add_argument(
    "--proportion_kept_data", type=str, default="{'fr': 0.05, 'en': 0.05, 'ar': 0.05}"
)
parser.add_argument("--trained_languages", type=str, default="['en', 'ar', 'fr']")
parser.add_argument("--method_similarity", type=str, default="embeddings")
parser.add_argument("--clustering_method", type=str, default="hdbscan")

args, _ = parser.parse_known_args()

languages = literal_eval(args.trained_languages)
proportions_kept_data = literal_eval(args.proportion_kept_data)


def clean_tweets(sentence, language: str):
    """
    function to clean tweets:
    1) remove links
    2) remove users
    3) lower and remove punctuation
    4) stem and lemmatize if english language
    """

    if type(sentence) is not str:
        sentence = str(sentence)

    # NB: the two functions below were taken from https://ourcodingclub.github.io/tutorials/topic-modelling-python/
    def remove_links(tweet):
        """Takes a string and removes web links from it"""
        tweet = re.sub(r"http", "", tweet)
        tweet = re.sub(r"http\S+", "", tweet)  # remove http links
        tweet = re.sub(r"bit.ly/\S+", "", tweet)  # rempve bitly links
        tweet = tweet.strip("[link]")  # remove [links]
        return tweet

    def remove_users(tweet):
        """Takes a string and removes retweet and @user information"""
        tweet = re.sub("(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)", "", tweet)  # remove retweet
        tweet = re.sub("(@[A-Za-z]+[A-Za-z0-9-_]+)", "", tweet)  # remove tweeted at
        return tweet

    new_words = []
    sentence = remove_links(sentence)
    sentence = remove_users(sentence)
    words = sentence.split()

    for word in words:

        # lower and remove punctuation
        new_word = re.sub(r"[^\w\s]", "", (word))

        # keep clean words and remove hyperlinks
        word_not_nothing = new_word != ""
        word_not_stop_word = new_word.lower() not in stop_words

        if word_not_nothing and word_not_stop_word:
            if language == "en":

                # lemmatize
                new_word = wordnet_lemmatizer.lemmatize(new_word, pos="v")

                # stem
                new_word = porter_stemmer.stem(new_word)

            new_words.append(new_word)

    return " ".join(new_words).rstrip().lstrip()


def get_embeddings(texts):
    """
    get all tweets embeddings, one embedding per tweet
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(texts)


def get_similarity_matrix(cleaned_tweets: List[str], method: str):
    """
    get similarity matrix according to 2 different methods
    INPUTS:
        - cleaned_tweets: list of preprocessed tweets
        - method: the method we are using. Can be 'tf-idf' or 'embeddings'
            - 'tf-idf': get tf-idf vectorizer of each tweet
            - 'embeddings': get embeddings of each tweet
    OUTPUTS:
        - similarity matrix of tweets
    """

    def similarity_using_tf_idf(cleaned_tweets: List[str]):
        # define and use tf-idf transformation
        tf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=0)
        tf_idf = tf.fit_transform(cleaned_tweets)

        # get cosine similarity matrix
        cosine_similarity_matrix = linear_kernel(tf_idf, tf_idf)
        return cosine_similarity_matrix

    if method == "tf-idf":
        return similarity_using_tf_idf(cleaned_tweets)
    elif method == "embeddings":
        embeddings_all_sentences = get_embeddings(cleaned_tweets)
        return cosine_similarity(embeddings_all_sentences, embeddings_all_sentences)
    else:
        return AssertionError(
            "wrong method name to get similarity matrix, please choose one in ['tf-idf', 'embeddings']"
        )


def get_louvain_partitions(
    df: pd.DataFrame, cleaned_tweets: List[str], method_similarity: str
):
    """
    function to get louvain partitions

    INPUT: List of preprocessed tweets
    OUTPUT: cluster label of each tweet

    1) get cosine simialrity of tweets
    2) create undirected graph where nodes are tweets and edges are the cosnie similarities
    3) get partitions using the louvain algorithm
    """
    original_tweets = df.tweet.tolist()
    tweet_ids = df.tweet_id.tolist()

    cosine_similarity_matrix = get_similarity_matrix(cleaned_tweets, method_similarity)

    # create graph from similarity matrix
    graph_one_lang = nx.Graph()
    matrix_shape = cosine_similarity_matrix.shape
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            # do only once
            if i < j:
                sim = cosine_similarity_matrix[i, j]
                graph_one_lang.add_edge(i, j, weight=sim)
                graph_one_lang.add_edge(j, i, weight=sim)

    # louvain community
    partition = community.best_partition(graph_one_lang)

    ids = []
    tweets = []
    partitions = []
    for key, val in partition.items():
        ids.append(tweet_ids[key])
        tweets.append(original_tweets[key])
        partitions.append(val)

    df_partition = pd.DataFrame(
        list(zip(ids, tweets, partitions)),
        columns=["tweet_id", "sentence", "partition"],
    ).sort_values(by="partition", inplace=False)

    return df_partition


def get_hdbscan_partitions(tweets: List[str]):
    """
    function to get HDBscan partitions: inspired from https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

    INPUT: List of preprocessed tweets
    OUTPUT: cluster label of each tweet

    1) get embeddings of tweets
    2) data reduction algorithm: UMAP
    3) HDBscan clustering
    """
    print("begin getitng embeddings")
    embeddings = get_embeddings(tweets)
    print("begin running umap")
    umap_embeddings = umap.UMAP(
        n_neighbors=10, n_components=8, metric="cosine"
    ).fit_transform(embeddings)
    print("begin running hdbscan")
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=5, metric="euclidean", cluster_selection_method="eom"
    ).fit(umap_embeddings)

    return cluster.labels_


def get_topics(cleaned_tweets: List[str]):
    """
    Topic modelling using LDA.
    To have a wide range of topics, we return for each cluster 2 topics with 5 features for each.
    """
    vectorizer = CountVectorizer(analyzer="word")
    tf = vectorizer.fit_transform(cleaned_tweets).toarray()

    number_of_topics = 2
    lda_model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)

    lda_model.fit(tf)
    feature_names = vectorizer.get_feature_names()

    topics = {}
    for topic_number, topic in enumerate(lda_model.components_):
        topics[topic_number] = [feature_names[i] for i in (topic.argsort()[-5:])]

    return str(topics)


def get_clusters_one_df(
    df: pd.DataFrame,
    cleaned_tweets: List[str],
    clustering_method: str,
    louvain_similarity_method: str = None,
):

    """
    main function to get clusters for one DataFrame.
    INPUTS:
        - df: original DataFrame
        - cleaned_tweets: List of preprocessed tweets.
        - clustering_method: method we use for clustering tweets: can be 'louvain' or 'hdbscan'.
        - louvain_similarity_method: if we use the 'louvain' clustering method,
        choose one method to compute simialrity between 'tf-idf' and 'embeddings'.

    OUTPUTS: DataFrame with clusters for each tweet
    """

    if clustering_method == "louvain":
        if louvain_similarity_method is None:
            AssertionError(
                "'louvain_similarity_method' not provided. Please choose one in ['tf-idf', 'embeddings] for computing similarity"
            )
        elif louvain_similarity_method is None:
            AssertionError("'language' not provided.")
        else:
            partitioned_df = get_louvain_partitions(
                df, cleaned_tweets, louvain_similarity_method
            )

    elif clustering_method == "hdbscan":

        partitioned_df = df.copy()
        partitioned_df["partition"] = get_hdbscan_partitions(cleaned_tweets)

    else:
        AssertionError(
            "Wrong clustering method name, please choose one of ['louvain', 'hdbscan']"
        )

    return partitioned_df


def get_clusters(
    df: pd.DataFrame,
    clustering_method: str,
    language: str,
    louvain_similarity_method: str = None,
):
    """
    Main function for clustering.....................
    INPUTS:
        -df: original DataFrame
        - clustering_method: method we use for clustering tweets: can be 'louvain' or 'hdbscan'.
        - language: one of ['en', 'fr, 'ar']
        - louvain_similarity_method: if we use the 'louvain' clustering method,
        choose one method to compute simialrity between 'tf-idf' and 'embeddings'.

    OUTPUTS:

    1) preprocess tweets
    2) get clustters
    3) get topics for tweets being in clusters
    """

    # preprocess tweets
    df["cleaned_tweets"] = [clean_tweets(one_tweet, language) for one_tweet in df.tweet]

    # get clustters
    partitioned_df = get_clusters_one_df(
        df.copy(),
        df["cleaned_tweets"].tolist(),
        clustering_method,
        louvain_similarity_method,
    )
    print("begin the topic modelling")

    clusters = partitioned_df.partition.unique()
    meaningful_clusters = clusters[clusters >= 0]

    final_df = partitioned_df[partitioned_df.partition == -1]
    final_df.loc[:, "topic"] = "UNKNOWN"
    final_df.loc[:, "mean_sentiment_score"] = final_df.overall_negative_sentiment

    # get topics for tweets being in clusters
    for cluster_tmp in tqdm(meaningful_clusters):
        df_one_cluster = partitioned_df[partitioned_df.partition == cluster_tmp]
        df_one_cluster.loc[
            :, "mean_sentiment_score"
        ] = df_one_cluster.overall_negative_sentiment.mean()
        df_one_cluster.loc[:, "topic"] = get_topics(
            df_one_cluster["cleaned_tweets"].tolist()
        )
        final_df = final_df.append(df_one_cluster)

    return final_df.drop(columns=["overall_negative_sentiment"], inplace=False)


def postprocess_df(df):
    """
    Function to postprocess DataFrame.
    Return dataframe sorted by number of tweets per topic.
    """
    df_copy = df.copy()

    df_partitions = (
        df_copy.groupby("partition", as_index=False)["tweet_id"]
        .apply(len)
        .rename(columns={"tweet_id": "cluster_size"})
    )
    df_partitions.index = df_partitions.partition
    df_partitions.drop(columns=["partition"], inplace=True)
    sorted_d = dict(
        sorted(
            df_partitions.to_dict()["cluster_size"].items(),
            key=operator.itemgetter(1),
            reverse=True,
        )
    )

    final_df = pd.DataFrame()
    for k, v in sorted_d.items():
        df_one_partition = df[df.partition == k]
        df_one_partition.loc[:, "cluster_size"] = v
        final_df = final_df.append(df_one_partition)

    return final_df


def get_relevant_hate_df(df: pd.DataFrame, n_kept_tweets: int):
    """
    filter out the relevant tweets: tweets with highest negative score.
    """
    if all([item in df.columns for item in ["offensive", "anger"]]):

        df["overall_negative_sentiment"] = df["anger"] + df["offensive"]

    final_df = df.sort_values(
        by="overall_negative_sentiment", inplace=False, ascending=False
    )[["tweet_id", "tweet", "overall_negative_sentiment"]].head(n_kept_tweets)

    return final_df


if __name__ == "__main__":

    print(
        "--------------------------------------------------------------------------------"
    )
    print(
        "---------------------- BEGIN RUNNING PRTITIONS SCRIPT --------------------------"
    )
    print(
        "--------------------------------------------------------------------------------"
    )

    for language_tmp in languages:

        print(f"----------------------- running for the {language_tmp} language")

        sentiments_df_one_language = pd.read_csv(
            f"generated_data/sentiments_numbers_{language_tmp}.csv", compression="gzip"
        )
        proportion_kept_data_one_lang = proportions_kept_data[language_tmp]
        n_tweets_one_language = len(sentiments_df_one_language)
        n_kept_tweets = int(n_tweets_one_language * proportion_kept_data_one_lang)

        relevant_hate_df = get_relevant_hate_df(
            sentiments_df_one_language, n_kept_tweets
        )

        df_one_language = get_clusters(
            df=relevant_hate_df,
            clustering_method=args.clustering_method,
            language=language_tmp,
            louvain_similarity_method=args.method_similarity,
        )

        final_df_one_language = postprocess_df(df_one_language)

        final_df_one_language.to_csv(
            f"generated_data/partitions_{language_tmp}_{args.clustering_method}_final.csv",
            index=None,
            compression="gzip",
        )

    print("--------------------------------------------------------------------")
    print("---------------------- SCRIPT RUN SUCCESSFULLY! --------------------")
    print("--------------------------------------------------------------------")
