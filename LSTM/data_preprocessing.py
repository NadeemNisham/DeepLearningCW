import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import string
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(file_path, chunk_size, num_samples):
    nltk.download('punkt')
    
    # Open the JSON file and process it in chunks with explicit encoding
    chunk_count = 0
    dataframe = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for chunk in pd.read_json(file, lines=True, chunksize=chunk_size):
            dataframe.append(chunk[['text', 'stars']].copy())
            chunk_count += 1

            if sum(map(len, dataframe)) >= num_samples:
                break

    yelp_subset = pd.concat(dataframe, ignore_index=True)
    output_json_file_path = 'yelp_subset.json'
    yelp_subset.to_json(output_json_file_path, orient='records', lines=True)

    # Display the shape of the loaded subset
    print(f"Shape of Yelp Subset after preprocessing: {yelp_subset.shape}")

    return yelp_subset
