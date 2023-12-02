from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, x_test_pad, y_test_one_hot, y_test_encoded, label_encoder, history):
    # Model Evaluation
    loss, accuracy = model.evaluate(x_test_pad, y_test_one_hot)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    # Calculate predictions
    y_pred = model.predict(x_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(np.array(y_test_one_hot), axis=1)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    # Confusion Matrix
    cm = confusion_matrix(y_test_encoded, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.show()

    # Plot Training & Validation Loss Values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot for Accuracy over Epochs
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Accuracy', marker='o', color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
