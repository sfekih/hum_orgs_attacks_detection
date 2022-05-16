from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import numpy as np
import pandas as pd
from scipy.special import softmax

import csv
import urllib.request
from tqdm import tqdm
import argparse
from typing import List
from ast import literal_eval


classification_tasks = ["offensive", "emotion"]
kept_sentiments = ["offensive", "anger"]

parser = argparse.ArgumentParser()

parser.add_argument("--use_sample", type=str, default="false")
parser.add_argument("--data_path", type=str, default="data/clean_data_march.csv")
parser.add_argument("--trained_languages", type=str, default="['en', 'ar', 'fr']")

args, _ = parser.parse_known_args()

languages = literal_eval(args.trained_languages)


def preprocess(text: str, process_tags: bool = True):
    """
    function to preprocess tags before feeding them to pretrained sentiment models.
    """
    new_text = []
    for t in str(text).split(" "):
        if process_tags:
            t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def get_sentiments_non_ar(df: pd.DataFrame):
    """
    function to get sentiments when the language is not arabic (french or english in our case)
    We work with 2 tasks here: 'offensive' and 'emotion'.
    """

    def get_sentiment_one_task(tweets_list: List[str], task: str):
        """
        get sentiments column for each different task
        """
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        labels = [row[1] for row in csvreader if len(row) > 1]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        tot_scores = []

        # get sentiment score for each tweet
        for tweet_tmp in tqdm(tweets_list):

            text = preprocess(tweet_tmp)
            encoded_input = tokenizer(text, return_tensors="pt")
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            tot_scores.append(scores)

        return tot_scores, labels

    def get_overall_negative_score(df):
        """
        normalize scores for each differeent sentiment (anger or offensiveness) then get mean for each row
        """
        sentiments_df = df.copy()

        # normalize scores
        for task in kept_sentiments:
            col_values = sentiments_df[task]
            mean = col_values.mean()
            std = col_values.std()
            sentiments_df[task] = (col_values - mean) / std

        return sentiments_df[kept_sentiments].mean(axis=1)

    data_df = df.copy()
    tweets_list = data_df.tweet.tolist()

    for task in classification_tasks:

        print(f"begin getting scores for {task}")

        tot_scores, labels = get_sentiment_one_task(tweets_list, task)
        data_df["scores"] = tot_scores
        n_labels = len(labels)

        for i in range(n_labels):
            label = labels[i]
            if label in kept_sentiments:
                data_df[label] = data_df["scores"].apply(lambda x: x[i])

        data_df.drop(columns=["scores"], inplace=True)

    # compute mean of three tasks
    data_df["overall_negative_sentiment"] = get_overall_negative_score(data_df)

    return data_df


def get_sentiments_ar(df: pd.DataFrame):
    """
    function to get sentiments when the language is arabic.
    """
    data_df = df.copy()
    tweets = data_df.tweet.tolist()
    sa = pipeline(
        "sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
    )

    model_returns = []
    batch_size = 32
    for i in tqdm(range(0, len(tweets), batch_size)):
        batch = tweets[i : i + batch_size]
        model_returns_one_batch = sa(batch)
        model_returns += model_returns_one_batch

    negative_scores = np.array(
        [
            one_return["score"]
            if one_return["label"] == "negative"
            else 1 - one_return["score"]
            for one_return in model_returns
        ]
    )
    mean = np.mean(negative_scores)
    std = np.std(negative_scores)

    data_df["overall_negative_sentiment"] = (negative_scores - mean) / std

    return data_df


def get_negative_sentiments(df: pd.DataFrame, language: str):

    if language == "ar":
        return get_sentiments_ar(df)
    else:
        return get_sentiments_non_ar(df)


if __name__ == "__main__":

    print(
        "---------------------------------------------------------------------------------"
    )
    print(
        "---------------------- BEGIN RUNNING SENTIMENTS SCRIPT --------------------------"
    )
    print(
        "---------------------------------------------------------------------------------"
    )

    data_df = pd.read_csv(args.data_path, compression="gzip")

    data_df["tweet"] = data_df.tweet.apply(lambda x: preprocess(x, process_tags=False))

    data_df.drop_duplicates(inplace=True)
    data_df["tweet_id"] = data_df.index

    if args.use_sample == "true":
        final_df = pd.DataFrame()
        for lang in ["en", "fr", "ar"]:
            final_df = final_df.append(
                data_df[data_df.language == lang].sample(frac=0.01)
            )

        data_df = final_df

    for language_tmp in languages:

        print(f"------------ begin getting sentiments for {language_tmp}")
        data_one_language = data_df[data_df.language == language_tmp]
        data_one_language.drop(columns="language", inplace=True)

        sentiments_df_one_language = get_negative_sentiments(
            data_one_language, language_tmp
        )

        sentiments_df_one_language.to_csv(
            f"generated_data/sentiments_numbers_{language_tmp}.csv",
            index=None,
            compression="gzip",
        )

    print("--------------------------------------------------------------------")
    print("---------------------- SCRIPT RUN SUCCESSFULLY! --------------------")
    print("--------------------------------------------------------------------")
