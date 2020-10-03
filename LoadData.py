import os
import pickle
import re

import pandas
import tensorflow
from keras_preprocessing.text import Tokenizer
import numpy as np
from keras_preprocessing.sequence import pad_sequences


def save_object(object, output_file_path):
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(object, output_file, pickle.HIGHEST_PROTOCOL)


def retrieve_object(input_file_path):
    with open(input_file_path, 'rb') as input_file:
        return pickle.load(input_file)


def clean(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)


class LoadData:
    def __init__(self, path, batch=32, split=0, text_processor=None):
        self.splitted_frame = None
        self.data_frame = pandas.read_csv(path)
        self.data_frame = self.data_frame.loc[:, ~self.data_frame.columns.str.contains('^Unnamed')]
        self.names = list(self.data_frame.columns)
        self.batch = batch
        self.set_split(split)
        self.text_processor = text_processor
        self.values = None

    def set_split(self, split=0.01):
        if 0 < split < 1:
            main_size = int(round(self.data_frame.size / self.data_frame.columns.size * (1 - split)))
            if main_size > 0:
                self.splitted_frame = self.data_frame[main_size:]
                self.data_frame = self.data_frame[:main_size]

    def train_tokenizer(self, columns: [], size: int = 1000):
        tokenizer = Tokenizer(size, oov_token='xxxxxxx')
        for column in columns:
            tokenizer.fit_on_texts(column)
        return tokenizer

    def execute_text_processor(self):
        if self.text_processor is None or "columns" not in self.text_processor:
            return None
        texts = []
        for column in self.text_processor["columns"]:
            self.data_frame[column] = self.data_frame[column].apply(clean)
            texts.append(self.data_frame[column])
            if self.splitted_frame is not None:
                self.splitted_frame[column] = self.splitted_frame[column].apply(clean)
                texts.append(self.splitted_frame[column])
        if "path" in self.text_processor and self.text_processor["path"] is not None and os.path.exists(
                self.text_processor["path"]):
            tokenizer = retrieve_object(self.text_processor["path"])
        else:
            num_words = self.text_processor["num_words"] if "num_words" in self.text_processor else 10000
            tokenizer = self.train_tokenizer(texts, num_words)
            if "path" in self.text_processor:
                save_object(tokenizer, self.text_processor["path"])
        values = {"main": {}, "secondary": {}}
        max_length = self.text_processor["max_length"] if "max_length" in self.text_processor else 100
        for column in self.text_processor["columns"]:
            values["main"][column] = pad_sequences(tokenizer.texts_to_sequences(self.data_frame[column]),
                                                   padding="post", maxlen=max_length)
            if self.splitted_frame is not None:
                values["main"][column] = pad_sequences(tokenizer.texts_to_sequences(self.splitted_frame[column]),
                                                       padding="post", maxlen=max_length)
        return values

    @property
    def data_set(self):
        if self.values is None:
            self.values = self.execute_text_processor()
        return self.to_data_set(self.data_frame, self.values["main"])

    @property
    def splited_data_set(self):
        if self.splitted_frame is not None:
            if self.values is None:
                self.values = self.execute_text_processor()
            return self.to_data_set(self.splitted_frame, self.values["secondary"])

    @property
    def x(self):
        features = self.data_frame[self.names[:-1]]
        return features.values

    @property
    def y(self):
        label = self.data_frame[self.names[-1]]
        return tensorflow.keras.utils.to_categorical(label.values)

    def to_data_set(self, frame, processed_values):
        label = frame[self.names[-1]]
        features = frame[self.names[:-1]]
        if processed_values is not None and len(processed_values) > 0:
            data = []
            non_processed = []
            for column in features.columns:
                if column in processed_values:
                    if len(non_processed) > 0:
                        data.append(features[non_processed].to_numpy())
                    data.append(processed_values[column])
                    non_processed = []
                else:
                    non_processed.append(column)
            if len(non_processed) > 0:
                data.append(features[non_processed].to_numpy())
            features_numpy = np.concatenate(data, 1) if len(data) > 1 else data[0]
        else:
            features_numpy = features.to_numpy()
        return tensorflow.data.Dataset.from_tensor_slices(
            (features_numpy, tensorflow.keras.utils.to_categorical(label.to_numpy()))).batch(self.batch)
