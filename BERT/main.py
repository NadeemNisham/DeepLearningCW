from data_preprocessing import preprocess_data
from feature_engineering import tokenize_data
from model_building import build_bert_model
from model_training import train_bert_model
from torch.utils.data import DataLoader, TensorDataset
import torch

def main():
    # Set the path to your Yelp dataset JSON file
    json_file_path = '../../yelp_academic_dataset_review.json'

    # Data Preprocessing
    x_train_resampled, x_test, y_train_resampled, y_test = preprocess_data(json_file_path)

    # Feature Engineering
    train_encodings, test_encodings = tokenize_data(x_train_resampled, x_test)

    # Model Building
    bert_model = build_bert_model()

    # Model Training
    train_loader = DataLoader(TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train_resampled.values)), batch_size=8, shuffle=True)
    model = train_bert_model(bert_model, train_loader)

    # Additional steps for model evaluation or inference can be added here

if __name__ == "__main__":
    main()
