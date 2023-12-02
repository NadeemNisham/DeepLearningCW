import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(json_file_path, chunk_size=5000, num_samples=20000):
    # Load a subset of the dataset
    dataframe = []
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for chunk in pd.read_json(file, lines=True, chunksize=chunk_size):
            dataframe.append(chunk[['text', 'stars']].copy())
            if sum(map(len, dataframe)) >= num_samples:
                break
    yelp_subset = pd.concat(dataframe, ignore_index=True)
    output_json_file_path = 'yelp_subset.json'
    yelp_subset.to_json(output_json_file_path, orient='records', lines=True)
    print("Shape of Yelp Subset:", yelp_subset.shape)

    # Derive sentiment labels from star labels
    yelp_subset['sentiment'] = yelp_subset['stars'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))

    # Handle imbalanced classes with random undersampling
    x_train, x_test, y_train, y_test = train_test_split(yelp_subset['text'], yelp_subset['sentiment'], test_size=0.2, random_state=42)
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    x_train_resampled, y_train_resampled = undersampler.fit_resample(yelp_subset['text'].to_frame(), yelp_subset['sentiment'])
    x_train_resampled = x_train_resampled['text']
    y_train_resampled = pd.Series(y_train_resampled)

    return x_train_resampled, x_test, y_train_resampled, y_test
