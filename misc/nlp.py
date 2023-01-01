from tensorflow import keras
import numpy as np
import os

def model(x, maxlen):
    x = keras.layers.Embedding(10000, 8, input_length=maxlen)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)
    return x

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = max_features)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

print('X_TRAIN, X_TEST SHAPE:', x_train.shape, x_test.shape)


inputs = keras.Input(shape=(maxlen))
outputs = model(inputs, maxlen)
model = keras.Model(inputs, outputs)

print('MODEL SHAPES:', inputs.shape, outputs.shape)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

