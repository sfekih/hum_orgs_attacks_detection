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

import umap
import hdbscan

import networkx as nx
import community.community_louvain as community
from ast import literal_eval
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--proportion_kept_data', type=str, default="{'fr': 0.1, 'en': 0.02, 'ar': 0.1}")
parser.add_argument('--trained_languages', type=str, default="['en', 'ar', 'fr']")
parser.add_argument('--method_similarity', type=str, default="embeddings")
parser.add_argument('--clustering_method', type=str, default='louvain')

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
        return AssertionError('wrong method name to get similarity matrix')

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
    embeddings = get_embeddings(tweets)
    umap_embeddings = umap.UMAP(n_neighbors=20,
                            n_components=6, 
                            metric='cosine').fit_transform(embeddings)

    cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

    return cluster.labels_


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
        kept_df = sentiments_df_one_language.sort_values(
            by='overall_negative_sentiment',
            ascending=False,
            inplace=False
            ).head(n_kept_tweets)

        if args.clustering_method=='louvain':
            partitioned_df = get_louvain_partitions(kept_df, language_tmp, args.method_similarity)
            partitioned_df.to_csv(
                f'generated_data/partitions_{language_tmp}_louvain_{args.method_similarity}.csv', 
                index=None, compression='gzip'
            )
        elif args.clustering_method=='hdbscan':
            partitioned_df = kept_df[['tweet_id', 'tweet']].rename(
                columns={'tweet': 'sentence'}
            ).copy()
            partitioned_df['partition'] = get_hdbscan_partitions(partitioned_df.tweet.tolist())
            partitioned_df.to_csv(
                f'generated_data/partitions_{language_tmp}_hdbscan.csv', 
                index=None, compression='gzip'
            )
        else: 
            AssertionError('Wrong clustering method name')

    print('--------------------------------------------------------------------')
    print('---------------------- SCRIPT RUN SUCCESSFULLY! --------------------')
    print('--------------------------------------------------------------------')