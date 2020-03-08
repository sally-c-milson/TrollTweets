from __future__ import print_function
import collections
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

data_path = "./data/"
run = "train"
tweet_max = 70
loadOld = False
use_dropout=False
first = True

def read_words(text):
    return bytes(text, 'utf-8').decode().replace("\n", "<eos>").split()


def build_vocab(rows):
    split = np.array([])
    for row in rows:
        split = np.append(split, read_words(row))

    counter = collections.Counter(split)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def tweet_to_word_ids(text, word_to_id):
    data = read_words(text)
    vec = []
    for word in data:
        vec.append(word_to_id[word] if word in word_to_id else len(word_to_id))
    return vec


def load_data(file, first):
    if first:
        # get the data paths
        path = os.path.join(data_path, file+'.csv')
        raw_data = pd.read_csv(path)
        raw_data = raw_data[['content', 'troll']]

        # build the complete vocabulary, then convert text data to list of integers
        # vocab built off of first 5000 tweets, this should contain most common words
        word_to_id = build_vocab(raw_data['content'][0:5000].to_numpy())
        vocabulary = len(word_to_id)
        reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

        processed_data = pd.DataFrame()
        processed_data['troll'] = raw_data['troll']
        processed_data['content'] = raw_data['content'].apply(lambda s: tweet_to_word_ids(s, word_to_id))

        # pad out to input length
        processed_data['content'] = processed_data['content'].apply(lambda x: np.pad(x, (0, tweet_max - len(x)%tweet_max), 'constant', constant_values=((vocabulary + 1), (vocabulary + 1))) if len(x) < tweet_max else x[0:tweet_max])

        print(processed_data.head())
        # print(word_to_id)
        print(vocabulary)

        x = processed_data['content'].to_numpy()
        y = processed_data['troll'].to_numpy()

        save_obj = {'x': x, 'y': y, 'vocabulary': vocabulary, 'reversed_dictionary': reversed_dictionary}
        outfile = open(os.path.join(data_path, file+'.pkl'),'wb')
        pickle.dump(save_obj, outfile)
        outfile.close()
    else:
        infile = open(os.path.join(data_path, file+'.pkl'),'rb')
        save_obj = pickle.load(infile)
        x = save_obj['x']
        y = save_obj['y']
        vocabulary = save_obj['vocabulary']
        reversed_dictionary = save_obj['reversed_dictionary']

    return x, y, vocabulary, reversed_dictionary


class KerasBatchGenerator(object):

    def __init__(self, x_data, y_data, batch_size, vocabulary):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0

    def generate(self):
        x = np.zeros((self.batch_size, tweet_max))
        y = np.zeros(self.batch_size)
        while True:
            for i in range(self.batch_size):
                if self.current_idx >= len(self.x_data):
                    self.current_idx = 0
                x[i, :] = np.array(self.x_data[self.current_idx])
                y[i] = 1 if self.y_data[self.current_idx] else 0
                self.current_idx += 1
            yield x, y


if __name__ == '__main__':
    X, y, vocabulary, reversed_dictionary = load_data('tweet_data_batch3', first)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, shuffle=True)

    if run == 'train':
        batch_size = 100
        train_data_generator = KerasBatchGenerator(X_train, y_train, batch_size, vocabulary)
        valid_data_generator = KerasBatchGenerator(X_valid, y_valid, batch_size, vocabulary)

        hidden_size = 500

        if loadOld:
            model = load_model(data_path + "/model-01.hdf5")
        else:
            model = Sequential()
            # +1 vocab for not in dict, +2 for padding
            model.add(Embedding(vocabulary + 2, hidden_size, input_length=tweet_max))
            model.add(LSTM(hidden_size, return_sequences=True))
            model.add(LSTM(hidden_size, return_sequences=False))
            if use_dropout:
                model.add(Dropout(0.5))

            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            optimizer = Adam()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
        num_epochs = 20
        # model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, validation_data=valid_data_generator.generate(), validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
        model.fit_generator(train_data_generator.generate(), 2000, num_epochs, validation_data=valid_data_generator.generate(), validation_steps=20, callbacks=[checkpointer])
        model.save(data_path + "final_model.hdf5")

    elif run == 'test':
        model = load_model(data_path + "/model-02.hdf5")
        dummy_iters = 40
        example_training_generator = KerasBatchGenerator(X_train, y_train, 1, vocabulary)
        print("Training data:")
        for i in range(dummy_iters):
            dummy = next(example_training_generator.generate())
        num_predict = 10
        for i in range(num_predict):
            data = next(example_training_generator.generate())
            prediction = model.predict(data[0])
            print(X_train[dummy_iters + i])
            print("predict: ", prediction, " real: ", y_train[dummy_iters + i])

        # test data set
        dummy_iters = 40
        example_test_generator = KerasBatchGenerator(X_valid, y_valid, 1, vocabulary)
        print("Test data:")
        for i in range(dummy_iters):
            dummy = next(example_test_generator.generate())
        num_predict = 10
        for i in range(num_predict):
            data = next(example_test_generator.generate())
            prediction = model.predict(data[0])
            print(X_valid[dummy_iters + i])
            print("predict: ", prediction, " real: ", y_valid[dummy_iters + i])

