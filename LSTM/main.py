from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model_building import build_model
from model_training import train_model
from model_evaluation import evaluate_model

# File path and other parameters
file_path = '../../yelp_academic_dataset_review.json'
chunk_size = 1000
num_samples = 50000

# Data Preprocessing
yelp_subset = preprocess_data(file_path, chunk_size, num_samples)

# Feature Engineering
max_num_words = 3000
embedding_dim = 50
max_len = 300
x_train_pad, y_train_one_hot, x_test_pad, y_test_one_hot, y_test_encoded, label_encoder = feature_engineering(yelp_subset)

# Model Building
model = build_model(max_num_words, embedding_dim, max_len)

# Model Training
model, history = train_model(model, x_train_pad, y_train_one_hot)

# Model Evaluation
evaluate_model(model, x_test_pad, y_test_one_hot, y_test_encoded, label_encoder, history)
