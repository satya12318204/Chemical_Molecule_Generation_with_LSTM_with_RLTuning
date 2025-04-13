import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ✅ Load Preprocessed Data
with open("/home/satya/Desktop/BI/processed_data/tokenized_data.pkl", "rb") as f:
    data = pickle.load(f)

tokenized_smiles = data["tokenized_smiles"]
char_to_idx = data["char_to_idx"]
idx_to_char = data["idx_to_char"]
max_smiles_length = data["max_smiles_length"]
num_tokens = len(char_to_idx)

# ✅ Prepare Sequences
X_train = [[char_to_idx.get(token, char_to_idx["<UNK>"]) for token in tokens] for tokens in tokenized_smiles]
X_train = pad_sequences(X_train, maxlen=max_smiles_length, padding="post")

X = X_train[:, :-1]  # All tokens except last
y = X_train[:, 1:]   # All tokens except first (next-token prediction)

# ✅ One-Hot Encode the Output
y_one_hot = to_categorical(y, num_classes=num_tokens)

# ✅ Define LSTM Model
def build_lstm_model(input_dim, output_dim, input_length):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=128, input_length=input_length),
        LSTM(256, return_sequences=True),
        LSTM(256, return_sequences=True),
        Dense(output_dim, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ✅ Initialize and Train Model
model = build_lstm_model(num_tokens, num_tokens, max_smiles_length - 1)
model.fit(X, y_one_hot, batch_size=64, epochs=50, validation_split=0.1)

# ✅ Save Trained Model
model_path = "/home/satya/Desktop/BI/saved_model/Orig/lstm_generator.h5"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"✅ Model saved at {model_path}")
