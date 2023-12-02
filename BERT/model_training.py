# model_training.py

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping
import torch

def train_bert_model(model, train_loader, epochs=5):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        for batch_num, batch in enumerate(train_loader):
            inputs = {'input_ids': batch[0].to(device),
                      'attention_mask': batch[1].to(device),
                      'labels': batch[2].to(device)}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
            true_labels.extend(inputs['labels'].cpu().numpy())

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}')

    model.save_pretrained('BERT_model')
    return model

