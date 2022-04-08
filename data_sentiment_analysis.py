from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

import numpy as np
import pandas as pd
from scipy.special import softmax

import json
import csv
import urllib.request
from tqdm import tqdm
import argparse
from typing import List

from create_sentence_paritions import get_louvain_partitions


classification_tasks = ['offensive', 'emotion', 'irony']
kept_sentiments = ['offensive', 'anger', 'irony']
languages = ['en', 'ar', 'fr']

parser = argparse.ArgumentParser()

parser.add_argument("--use_sample", type=str, default='false')
parser.add_argument('--data_path', type=str, default='data/clean_data_march.csv')
parser.add_argument('--proportion_kept_data', type=float, default=0.1)

args, _ = parser.parse_known_args()


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def get_sentiment_one_task(tweets_list: List[str], task: str):
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # download label mapping
    mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode('utf-8').split("\n")
        csvreader = csv.reader(html, delimiter='\t')
    labels = [row[1] for row in csvreader if len(row) > 1]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tot_scores = []

    for tweet_tmp in tqdm(tweets_list):

        text = preprocess(tweet_tmp)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        tot_scores.append(scores)

    return tot_scores, labels

def get_negative_sentiments(df: pd.DataFrame):

    tweets_list = data_one_language.tweet.tolist()
    data_df = df.copy()

    for task in classification_tasks:

        tot_scores, labels = get_sentiment_one_task(tweets_list, task)

        data_df['scores'] = tot_scores

        n_labels = len(labels)

        for i in range (n_labels):
            label = labels[i]
            if label in kept_sentiments:
                data_df[label] = data_df['scores'].apply(lambda x: x[i])
        
        data_df.drop(columns=['scores'], inplace=True)

    return data_df

def get_overall_negative_score(df):

    sentiments_df = df.copy()

    #normalize scores
    for task in kept_sentiments:
        col_values = sentiments_df[task]
        sentiments_df[task] = col_values.mean() / col_values.std()

    sentiments_df['overall_score']= sentiments_df[
            kept_sentiments
            ].mean(axis=1)

    return sentiments_df['overall_score']

if __name__ == '__main__':

    data_df = pd.read_csv(args.data_path, compression='gzip').drop_duplicates(inplace=False)
    
    if args.use_sample == 'true':
        final_df = pd.DataFrame()
        for lang in ['en', 'fr', 'ar']:
            final_df = final_df.append(data_df[data_df.language==lang].sample(frac=0.05))

        data_df = final_df
    

    for language_tmp in languages:
        data_one_language = data_df[data_df.language==language_tmp]
        n_tweets_one_language = len(data_one_language)

        sentiments_df_one_language = get_negative_sentiments(data_one_language)

        # compute mean of three tasks
        sentiments_df_one_language['overall_negative_sentiment'] = get_overall_negative_score(sentiments_df_one_language)
        
        sentiments_df_one_language.to_csv(
            f'generated_data/sentiments_numbers_{language_tmp}.csv', index=None, compression='gzip'
            )

        n_kept_tweets = int(n_tweets_one_language * args.proportion_kept_data)
        kept_df = sentiments_df_one_language.sort_values(
            by='overall_negative_sentiment',
            ascending=False,
            inplace=False
            ).head(n_kept_tweets)

        partitioned_df = get_louvain_partitions(kept_df)

        partitioned_df.to_csv(
            f'generated_data/partitions_{language_tmp}.csv', index=None, compression='gzip'
        )