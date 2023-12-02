from transformers import DistilBertForSequenceClassification

def build_bert_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    return model
