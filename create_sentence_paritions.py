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



def clean_tweets(row):
    sentence = row['tweet']
    language = row['language']

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_sample", type=str, default='false')
    parser.add_argument('--data_path', type=str, default='data/clean_data_march.csv')

    args, _ = parser.parse_known_args()

    data_original = pd.read_csv(args.data_path).drop_duplicates(inplace=False)

    if args.use_sample == 'true':
        data_original = data_original.head()

    treated_languages = ['en', 'ar', 'fr']

    data_original['cleaned_tweet'] = data_original.apply(lambda x: clean_tweets(x), axis=1)

    for lang in tqdm(treated_languages[::-1]) :

        # keep only needed language
        df_tmp = data_original[data_original['language'] == lang].drop_duplicates(inplace=False)
        if lang=='en':
            df_tmp = df_tmp.sample(frac=0.3)
        clean_data_one_lang = df_tmp.cleaned_tweet.tolist()
        original_data_one_lang = df_tmp.tweet.tolist()

        #define and use tf-idf transformation
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0)
        tf_idf = tf.fit_transform(clean_data_one_lang)

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
        partitioned_sentences = {original_data_one_lang[key]: val for key, val in partition.items()}

        df_partitions = pd.DataFrame(
            list(zip(
                list(partitioned_sentences.keys()), 
                list(partitioned_sentences.values())
                )),
            
            columns=['data', 'partition']
        ).sort_values(by='partition', inplace=False)

        df_partitions.to_csv(f'march_data_partitions_{lang}.csv', index=None)
    
