from typing import List
import numpy as np
import pandas as pd

import re
import json

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer  
porter_stemmer = PorterStemmer()
stop_words = set(stopwords.words())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sentence_transformers import SentenceTransformer

import umap.umap_ as umap
import hdbscan

import networkx as nx
import community.community_louvain as community
from ast import literal_eval
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--proportion_kept_data', type=str, default="{'fr': 0.05, 'en': 0.05, 'ar': 0.05}")
parser.add_argument('--trained_languages', type=str, default="['en', 'ar', 'fr']")
parser.add_argument('--method_similarity', type=str, default="embeddings")
parser.add_argument('--clustering_method', type=str, default='hdbscan')

args, _ = parser.parse_known_args()

languages = literal_eval(args.trained_languages)
proportions_kept_data = literal_eval(args.proportion_kept_data)

def clean_tweets(sentence, language: str):

    if type(sentence) is not str:
        sentence = str(sentence)

    new_words = []
    words = sentence.split()
    for word in words:
        
        #lower and remove punctuation
        new_word = re.sub(r'[^\w\s]', '', (word))

        #keep clean words and remove hyperlinks
        word_not_nothing = new_word != ''
        word_not_stop_word = new_word.lower() not in stop_words
        word_not_hyperlink = 'https' not in new_word
        #word_not_digit = ~new_word.isdigit()

        if word_not_nothing and word_not_stop_word and word_not_hyperlink:
            if language=='en':

                #lemmatize
                new_word =  wordnet_lemmatizer.lemmatize(new_word, pos="v")  

                #stem
                new_word = porter_stemmer.stem(new_word)

            new_words.append(new_word)
            
    return ' '.join(new_words)

def get_embeddings(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(texts)   

def get_similarity_matrix(cleaned_tweets: List[str], method: str):

    def similarity_using_tf_idf(cleaned_tweets: List[str]):
        #define and use tf-idf transformation
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
        tf_idf = tf.fit_transform(cleaned_tweets)

        # get cosine similarity matrix
        cosine_similarity_matrix = linear_kernel(tf_idf, tf_idf)
        return cosine_similarity_matrix 

    if method=='tf-idf':
        return similarity_using_tf_idf(cleaned_tweets)
    elif method=='embeddings':
        embeddings_all_sentences = get_embeddings(cleaned_tweets)
        return cosine_similarity(embeddings_all_sentences, embeddings_all_sentences)
    else:
        return AssertionError("wrong method name to get similarity matrix, please choose one in ['tf-idf', 'embeddings]")

def get_louvain_partitions(df: pd.DataFrame, language: str, method_similarity: str):

    original_tweets = df.tweet.tolist() 
    tweet_ids = df.tweet_id.tolist() 
    cleaned_tweets = [clean_tweets(one_tweet, language) for one_tweet in original_tweets] 

    cosine_similarity_matrix = get_similarity_matrix(cleaned_tweets, method_similarity)

    # create graph from similarity matrix
    graph_one_lang = nx.Graph()
    matrix_shape = cosine_similarity_matrix.shape
    for i in range (matrix_shape[0]):
        for j in range (matrix_shape[1]):
            #do only once
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
        list(zip(
            ids, 
            tweets,
            partitions
            )),
        
        columns=['tweet_id', 'sentence', 'partition']
    ).sort_values(by='partition', inplace=False)

    return df_partition

def get_hdbscan_partitions(tweets: List[str]):
    print('begin getitng embeddings')
    embeddings = get_embeddings(tweets)
    print('begin running umap')
    umap_embeddings = umap.UMAP(n_neighbors=10,
                            n_components=6, 
                            metric='cosine').fit_transform(embeddings)
    print('begin running hdbscan')
    cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

    return cluster.labels_


def get_clusters(
    df: pd.DataFrame, 
    clustering_method: str,
    language: str = None, 
    louvain_similarity_method: str = None
    ):

    labels = df.label.unique()
    for label in labels:
        get_clusters_one_df(
            df, 
            clustering_method,
            label,
            language, 
            louvain_similarity_method
        )

def get_clusters_one_df(
    df: pd.DataFrame, 
    clustering_method: str,
    sentiment_column: str,
    language: str = None, 
    louvain_similarity_method: str = None
    ):

    """if clustering_method not in ['hdbscan', 'louvain']: 
        AssertionError("Wrong clustering method name, please choose one of ['louvain', 'hdbscan']")"""

    if clustering_method=='louvain':
        if louvain_similarity_method is None:
            AssertionError("'louvain_similarity_method' not provided. Please choose one in ['tf-idf', 'embeddings] for computing similarity") 
        elif louvain_similarity_method is None:
            AssertionError("'language' not provided.") 
        else:
            partitioned_df = get_louvain_partitions(df, language, louvain_similarity_method)
           
    elif clustering_method=='hdbscan':

        partitioned_df = df.copy()
        partitioned_df['partition'] = get_hdbscan_partitions(partitioned_df.tweet.tolist())

    else: 
        AssertionError("Wrong clustering method name, please choose one of ['louvain', 'hdbscan']")

    partitioned_df.to_csv(
            f'generated_data/partitions_{language_tmp}_{clustering_method}_{sentiment_column}.csv', 
            index=None, compression='gzip'
        )

def get_relevant_hate_df(df: pd.DataFrame, n_kept_tweets: int):

    if all([item in df.columns for item in ['offensive', 'anger', 'irony']]):
        df['anger_offensive_score'] = df['anger'] + df['offensive']
        anger_offensive_df = df.sort_values(by='anger_offensive_score', inplace=False, ascending=False)[
            ['tweet_id', 'tweet']
        ].head(n_kept_tweets)
        anger_offensive_df['label'] = 'anger_offensive'

        irony_df = df.sort_values(by='irony', inplace=False, ascending=False)[
            ['tweet_id', 'tweet']
        ].head(n_kept_tweets)
        irony_df['label'] = 'irony'

        final_df = pd.concat([anger_offensive_df, irony_df]).drop_duplicates(inplace=False)

    else:
        final_df = df.sort_values(by='overall_negative_sentiment', inplace=False, ascending=False)[
            ['tweet_id', 'tweet']
        ].head(n_kept_tweets)
        final_df['label'] = 'overall_negative_score'

    return final_df

if __name__ == '__main__':

    print('--------------------------------------------------------------------------------')
    print('---------------------- BEGIN RUNNING PRTITIONS SCRIPT --------------------------')
    print('--------------------------------------------------------------------------------')

    for language_tmp in languages:

        print(f'running for the {language_tmp} language')

        sentiments_df_one_language = pd.read_csv(
            f'generated_data/sentiments_numbers_{language_tmp}.csv', 
            compression='gzip'
            )
        proportion_kept_data_one_lang = proportions_kept_data[language_tmp]
        n_tweets_one_language = len(sentiments_df_one_language)
        n_kept_tweets = int(n_tweets_one_language * proportion_kept_data_one_lang)

        relevant_hate_df = get_relevant_hate_df(sentiments_df_one_language, n_kept_tweets)

        get_clusters(
            df=relevant_hate_df, 
            clustering_method=args.clustering_method,
            language=language_tmp, 
            louvain_similarity_method=args.method_similarity
            )

    print('--------------------------------------------------------------------')
    print('---------------------- SCRIPT RUN SUCCESSFULLY! --------------------')
    print('--------------------------------------------------------------------')