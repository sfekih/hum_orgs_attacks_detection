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

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

if __name__=='main':

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_sample", type=str, default='false')
    parser.add_argument('--data_path', type=str, default='data/clean_data_march.csv')

    args, _ = parser.parse_known_args()

    data_df = pd.read_csv(args.data_path).drop_duplicates(inplace=False)

    if args.use_sample == 'true':
        data_df = data_df.head()

    data_original = data_df.tweet.tolist()

    tasks = ['offensive', 'emotion', 'irony']

    for task in tqdm(tasks):

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

        for tweet in data_original:

            text = preprocess(tweet)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            tot_scores.append(scores)

        data_df['scores'] = tot_scores

        for i in range (len(labels)):
            label = labels[i]
            data_df[label] = data_df['score'].apply(lambda x: x[i])
        
        data_df.drop(columns=['scores'], inplace=True)
        data_df.to_csv(f'sentiments_analysis.csv', index=None)
