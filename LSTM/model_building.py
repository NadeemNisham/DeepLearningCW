from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

def build_model(max_num_words, embedding_dim, max_len):
    # Model Building
    embedding_dim = 50
    model = Sequential()
    model.add(Embedding(input_dim=max_num_words, output_dim=embedding_dim, input_length=max_len))
    model.add(LSTM(100, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
