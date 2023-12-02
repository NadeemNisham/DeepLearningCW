from transformers import DistilBertTokenizer
import torch

def tokenize_data(x_train_resampled, x_test):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

    train_encodings = tokenizer(list(x_train_resampled), truncation=True, padding=True, max_length=400, return_tensors='pt')
    test_encodings = tokenizer(list(x_test), truncation=True, padding=True, max_length=400, return_tensors='pt')

    return train_encodings, test_encodings
