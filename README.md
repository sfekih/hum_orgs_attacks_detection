# Selim Fekih
### This project's purpose is to Detect tweet based attacks against humanitarian organisations (ICRC more specifically).

## project structure
```
├── generated_data                         # folder containg the generated data (all files are `gzip` compressed)
|   |__ final_icrc_hateful.csv             # Final generated tweet based attacks df agsinst ICRC
|   |__ partitions_en_hdbscan_final.csv    # HDBscan clusters for English
|   |__ partitions_fr_hdbscan_final.csv    # HDBscan clusters for French
|   |__ partitions_ar_hdbscan_final.csv    # HDBscan clusters for Arabic
|   |__ sentiments_numbers_en.csv          # sentiment results for English
|   |__ sentiments_numbers_fr.csv          # sentiment results for French
|   |__ sentiments_numbers_ar.csv          # sentiment results for Arabic
├── tweets_EDA.ipynb                       # Notebook containing Exploratory Data Analysis of tweets
├── data_sentiment_analysis.py             # Script to get general negative sentiment from tweets
├── analyze_sentiment_results.ipynb    	   # Notebook to analyze general results of sentiments scores
├── create_sentence_partitions.py          # Script to filter out then cluster negative tweets
├── decompress_files.py                    # Script to decompress the files from the `generated_data` folder
├── get_relevant_tweets.ipynb              # Notebook to analyze clusters and return final relevant file
├── requirements.txt                       # requirements to be installed before running the project
├── report.pdf                             # Contains the report of the project
└── README.md
```
## Procedure to run the project:
### 1) Add data: 
Add the original tweeter DataFrame as a new file `data/clean_data.csv` and `gzip` compressed
### 2) Run sentiments model:
```
python data_sentiment_analysis.py
  --use_sample: str     If 'true' then run with only a sample (1%) of each language, otherwise run for all tweets
  --data_path: str      data path of original data
  --trained_languages   languages trained on
```
### 3) Create sentence partitions to get final clusters
```
python create_sentence_partitions.py
  --proportion_kept_data        Prportion of data to be kept for each language
  --trained_languages           Languages we want to work on
  --clustering_method           One of 'louvain' and 'hdbscan'
  --method_similarity           If we use the 'louvain' clustering method, one of 'tf-idf' and 'embeddings'
```
### 4) Decompress generated files for visualization of results
```
python train.py
  --decompressed_folder_name: str     name of decompressed folder
```
### 5) Generate final results
Run the notebook `get_relevant_tweets.ipynb`