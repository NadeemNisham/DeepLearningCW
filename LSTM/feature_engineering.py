from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import string

def feature_engineering(yelp_subset):
    nltk.download('stopwords')

    # Feature Engineering
    # Derive sentiment labels from star labels
    yelp_subset['sentiment'] = yelp_subset['stars'].apply(lambda x: 'positive' if x > 3 else ('negative' if x < 3 else 'neutral'))

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(yelp_subset['text'], yelp_subset['sentiment'], test_size=0.2, random_state=42)

    # Convert all text to lowercase for uniformity
    x_train = x_train.apply(lambda x: x.lower())
    x_test = x_test.apply(lambda x: x.lower())

    # Remove punctuation to reduce noise in the text
    x_train = x_train.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    x_test = x_test.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Remove common words that may not contribute much to sentiment
    stop_words = set(stopwords.words('english'))
    x_train = x_train.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    x_test = x_test.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

    # Reduce words to the base form
    lemmatizer = WordNetLemmatizer()
    x_train = x_train.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
    x_test = x_test.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    # Reshape x_train to have two dimensions
    x_train_reshape = x_train.values.reshape(-1, 1)

    # Oversample the minority classes
    over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    x_train_over_resampled, y_train_resampled = over_sampler.fit_resample(x_train_reshape, y_train)

    # Convert oversampled indices to text sequences
    x_train_over = x_train_over_resampled.flatten()

    # Tokenization
    max_num_words = 3000
    tokenizer = Tokenizer(num_words=max_num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(x_train_over)
    x_train_seq = tokenizer.texts_to_sequences(x_train_over)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    # Define maximum length of the sequence
    max_len = 300

    # Pad sequences to make sentences the same length
    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len)
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len)

    # Label Encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_resampled)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert numerical lables to one-hot encoding
    y_train_one_hot = pd.get_dummies(y_train_encoded)
    y_test_one_hot = pd.get_dummies(y_test_encoded)

    return x_train_pad, y_train_one_hot, x_test_pad, y_test_one_hot, y_test_encoded, label_encoder

