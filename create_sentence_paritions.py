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
from sklearn.metrics.pairwise import linear_kernel

import networkx as nx
import community.community_louvain as community
from ast import literal_eval
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--proportion_kept_data', type=float, default=0.1)
parser.add_argument('--trained_languages', type=str, default="['en', 'ar', 'fr']")

args, _ = parser.parse_known_args()

languages = literal_eval(args.trained_languages)

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


def get_louvain_partitions(df: pd.DataFrame, language: str):

    original_tweets = df.tweet.tolist() 
    tweet_ids = df.tweet_id.tolist() 
    cleaned_tweet = [clean_tweets(one_tweet, language) for one_tweet in original_tweets] 

    #define and use tf-idf transformation
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
    tf_idf = tf.fit_transform(cleaned_tweet)

    # get cosine similarity matrix
    cosine_similarity_matrix = linear_kernel(tf_idf, tf_idf)

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

if __name__ == '__main__':

    print('--------------------------------------------------------------------------------')
    print('---------------------- BEGIN RUNNING PRTITIONS SCRIPT --------------------------')
    print('--------------------------------------------------------------------------------')

    for language_tmp in languages:

        sentiments_df_one_language = pd.read_csv(
            f'generated_data/sentiments_numbers_{language_tmp}.csv', 
            compression='gzip'
            )

        n_tweets_one_language = len(sentiments_df_one_language)
        n_kept_tweets = int(n_tweets_one_language * args.proportion_kept_data)
        kept_df = sentiments_df_one_language.sort_values(
            by='overall_negative_sentiment',
            ascending=False,
            inplace=False
            ).head(n_kept_tweets)

        partitioned_df = get_louvain_partitions(kept_df, language_tmp)

        partitioned_df.to_csv(
            f'generated_data/partitions_{language_tmp}.csv', index=None, compression='gzip'
        )

    print('--------------------------------------------------------------------')
    print('---------------------- SCRIPT RUN SUCCESSFULLY! --------------------')
    print('--------------------------------------------------------------------')