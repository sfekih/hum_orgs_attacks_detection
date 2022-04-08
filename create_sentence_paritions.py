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
from tqdm import tqdm
import argparse



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


def get_louvain_partitions(df):

    original_tweets = df.tweet.tolist() 
    language = df.language

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

    partitioned_sentences = {original_tweets[key]: val for key, val in partition.items()}

    df_partition = pd.DataFrame(
        list(zip(
            list(partitioned_sentences.keys()), 
            list(partitioned_sentences.values())
            )),
        
        columns=['tweet_id', 'partition']
    ).sort_values(by='partition', inplace=False)

    return df_partition