import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, Flatten

from LoadData import LoadData

if __name__ == '__main__':
    data = LoadData(
        "data/product_reviews_dataset.csv",
        32,
        0,
        {"path": "model",   "columns": ["Summary"]}
    )

    text_model = Sequential([
        Embedding(input_length=100, input_dim=10000, output_dim=50),
        Flatten(),
        Dense(6, activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    text_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")
    text_model.fit(data.data_set)
