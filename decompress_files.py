import os
import argparse
import pandas as pd

pd.options.mode.chained_assignment = None

from ast import literal_eval
import operator


parser = argparse.ArgumentParser()

parser.add_argument(
    "--decompressed_folder_name", type=str, default="generated_data_decompressed"
)
args, _ = parser.parse_known_args()
decompressed_folder_name = args.decompressed_folder_name


def custom_eval(x):
    """apply literal evaluation to topics"""
    if x == "UNKNOWN":
        return {"x": []}
    elif type(x) is not str:
        return x
    else:
        return literal_eval(x)


def flatten(t):
    """flatten list of lists"""
    return [item for sublist in t for item in sublist]


def clean_topic(topics):
    """keep all features from clusters after getting the topics"""
    return list(set(flatten(list(custom_eval(topics).values()))))


if __name__ == "__main__":

    if not os.path.exists(decompressed_folder_name):
        os.makedirs(decompressed_folder_name)

    files = os.listdir("generated_data")

    for file in files:
        print(f"begin decompressing the file: {file}")

        df_one_file = pd.read_csv(f"generated_data/{file}", compression="gzip")

        if "topic" in df_one_file:
            df_one_file["topic"] = df_one_file.topic.apply(clean_topic)

        df_one_file.to_csv(f"{decompressed_folder_name}/{file}", index=None)

    print("SCRIPT RUN SUCCESSFULLLY!")
