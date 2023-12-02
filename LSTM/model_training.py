from keras.callbacks import EarlyStopping

def train_model(model, x_train_pad, y_train_one_hot):
    # Model Training
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x_train_pad,
        y_train_one_hot,
        epochs=60,
        batch_size=500,
        validation_split=0.1,
        callbacks=[early_stopping]  # Add the early stopping callback
    )

    return model, history
