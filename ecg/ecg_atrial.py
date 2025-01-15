import keras
from tensorflow import reshape, concat
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Activation, MaxPool1D, Layer
from tensorflow.keras.models import Model
import tensorflow as tf

import config

class Concat(Layer):
    def call(self, x):
        return tf.concat(x, axis=-1)

class DDNN:
    def __init__(self):
        self.kernel_size_1 = 8
        self.kernel_size_2 = 16
        self.kernel_size_3 = 24
        self.name = "DDNN"

    # def __del__(self):
    #     print(self.name + " object disappears.")

    def seBlock(self, x):
        x1 = GlobalAveragePooling1D()(x)  # (1, 36)

        x1 = Dense(3)(x1)
        x1 = Activation("relu")(x1)
        x1 = Dense(x.shape[-1])(x1)
        x1 = Activation("sigmoid")(x1)
        x1 = reshape(x1, [-1, 1, x1.shape[-1]])
        return x * x1

    def transition(self, x):
        x = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x)
        x = MaxPool1D()(x)
        return x

    def ddnn(self):
        input = Input(shape=(config.FS * config.DURATION, 1))

        x8 = Conv1D(filters=1, kernel_size=self.kernel_size_1, padding="same")(input)
        x16 = Conv1D(filters=1, kernel_size=self.kernel_size_2, padding="same")(input)
        x24 = Conv1D(filters=1, kernel_size=self.kernel_size_3, padding="same")(input)

        # 5000, 36
        concat_layer = Concat()
        x = concat_layer([x8, x16])
        x = concat_layer([x, x24])

        # 1
        x = self.seBlock(x)
        x1 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x)
        x2 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1)
        x = x + x1 + x2
        x = self.seBlock(x)
        x = self.transition(x)

        # 2
        x = self.seBlock(x)
        x1 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x)
        x2 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1)
        x3 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2)
        x4 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2 + x3)
        x = x + x1 + x2 + x3 + x4
        x = self.seBlock(x)
        x = self.transition(x)

        # 3
        x = self.seBlock(x)
        x1 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x)
        x2 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1)
        x3 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2)
        x4 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2 + x3)
        x5 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2 + x3 + x4)
        x6 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2 + x3 + x4 + x5)
        x = x + x1 + x2 + x3 + x4 + x5 + x6
        x = self.seBlock(x)
        x = self.transition(x)

        # 4
        x1 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x)
        x2 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1)
        x3 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2)
        x4 = Conv1D(filters=x.shape[-1], kernel_size=self.kernel_size_2, padding="same")(x + x1 + x2 + x3)
        x = x + x1 + x2 + x3 + x4

        x = GlobalAveragePooling1D()(x)  # (1, 36) or (1, 3)
        x = Dense(1, activation='sigmoid')(x)
        cai_model = Model(inputs=input, outputs=x)
        return cai_model
