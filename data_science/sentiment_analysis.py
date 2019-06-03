#!/usr/bin/env python3

import argparse
import os
import sys
from contextlib import redirect_stderr
with redirect_stderr(open(os.devnull, "w")):
    os.environ['KERAS_BACKEND'] = 'theano'
    from keras.preprocessing import sequence
    from keras.models import Sequential
    from keras.layers import Dense, Embedding
    from keras.layers import LSTM
    from keras.datasets import imdb
    from keras.layers import Dropout, Activation
    from keras.layers import Conv1D, MaxPooling1D,BatchNormalization,Flatten
    from keras.models import load_model
    from keras.preprocessing.text import text_to_word_sequence

class model:
    def __init__(self, max_features = 100000,
                max_len = 100):
        self.max_features = max_features
        self.max_len = max_len
        self.words = imdb.get_word_index()

    def build(self):
        '''
        Build a CNN model
        '''
        max_features = 10000
        embedding_dims = 32
        filters = 128
        kernel_size = 3
        hidden_dims = 250

        epochs = 2
        number_conv_layer = 1

        self.model = Sequential()
        self.model.add(Embedding(self.max_features,
                        embedding_dims,
                        input_length=100))

        self.model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))

        # Flatten
        self.model.add(Flatten())

        # We add a vanilla hidden layers
        self.model.add(Dense(36))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))
        self.model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

        return model

    def train(self, x_train, y_train,
              x_test, y_test,
              batch_size = 32,
              epochs = 10,
              verbose = 2):
        '''
        Train model and return history
        '''
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose = 2)
        return history

    def save_model(self, file_name):

        self.model.save(file_name)

    def load_model(self, file_name):
        '''
        Load a previously trained model
        '''
        try:
            del self.model
        except: pass

        self.model = load_model(file_name)

    def summary(self):
        '''
        Return model summary
        '''
        self.model.summary()

    def _scores(self, results):

        final = []
        results = [item for sublist in results for item in sublist]
        for result in results:
            p = result * 2
            final.append(p-1)
        return final

    def predict_array(self, array):
        self.prediction = self._scores(self.model.predict(array))

    def predict_phrase(self, phrase):
        # define the document
        # tokenize the document

        word_vecs = []

        if isinstance(phrase, str):
            texts = [phrase[0:100]]

        if isinstance(phrase, list):
            texts = [x[0:100] for x in phrase]

        self.texts = texts
        for text in self.texts:
            result = text_to_word_sequence(text)
            word_vec = []
            for word in result:
                word_vec.append(self.words[word])

            word_vecs.append(word_vec)
        x = sequence.pad_sequences(word_vecs, maxlen= self.max_len)
        self.prediction = self._scores(self.model.predict(x))

def parse_cli():
    parser = argparse.ArgumentParser(description = 'Simple sentimental analysis')

    parser.add_argument('phrases'
                        )

    args = parser.parse_args()
    return args

def check_ipt(ipt):
    if os.path.isfile(ipt):
        return open(ipt).read().splitlines()
    else:
        return [ipt]

if __name__ == '__main__':

    args = parse_cli()
    ipt = check_ipt(args.phrases)
    # Load model
    m = model()
    m.load_model('./sentiment_analysis.h5')
    m.predict_phrase(ipt)

    print('Score\tphrase')
    for a,b in zip(ipt, m.prediction):
        print(f'{b:.2f}\t{a}')
