import re

import pandas
import tensorflow
from tensorflow.python import feature_column


class LoadData:
    def __init__(self, path, batch=32, split=0):
        self.splited_frame = None
        self.data_frame = pandas.read_csv(path)
        self.names = list(self.data_frame.columns)
        self.batch = batch
        self.set_split(split)
        # for index in range(0, len(self.names)):
        #     self.names[index] = ''.join(
        #         char for char in self.names[index] if re.match("[A-Za-z0-9_.\\-/]", char) is not None)
        # self.data_frame.columns = self.names

    def set_split(self, split=0.01):
        if 0 < split < 1:
            main_size = int(round(self.data_frame.size / self.data_frame.columns.size * (1 - split)))
            if main_size > 0:
                self.splited_frame = self.data_frame[main_size:]
                self.data_frame = self.data_frame[:main_size]

    @property
    def data_set(self):
        return self.to_data_set(self.data_frame)

    @property
    def splited_data_set(self):
        if self.splited_frame is not None:
            return self.to_data_set(self.splited_frame)

    @property
    def x(self):
        features = self.data_frame[self.names[:-1]]
        return features.values

    @property
    def y(self):
        label = self.data_frame[self.names[-1]]
        return tensorflow.keras.utils.to_categorical(label.values)

    def to_data_set(self, frame):
        label = frame[self.names[-1]]
        features = frame[self.names[:-1]]
        return tensorflow.data.Dataset.from_tensor_slices(
            (features.values, tensorflow.keras.utils.to_categorical(label.values))).batch(self.batch)
