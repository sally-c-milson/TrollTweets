import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec
import gensim.downloader as gensim

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

path = "./data/"
run = "train"
loadOld = True
use_dropout = True
first = False
loadTokenizer = True

MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 1000000
EMBEDDING_DIM = 200

def read_words(text):
    return bytes(text, 'utf-8').decode().replace("\n", "<eos>").lower().split()

def get_embedding():
    with open(path+"tweet_data_batch_all.pkl",'rb') as target:
        allData = pickle.load(target)
        embedding_matrix = allData['embedding_matrix']
        print("Loaded embedding matrix")
    return embedding_matrix

def load_data(file, first):
    if first:
        raw = pd.DataFrame()
        if loadTokenizer == True:
            embedding_matrix = get_embedding()
            raw = pd.read_csv(path+file+".csv")
            print(len(raw))
        else:
            # exclude file 0 for test
            for i in range(12):
                new = pd.read_csv(path+file+str(i+1)+".csv")
                new = new.set_index('tweet_id')
                raw = pd.concat([raw,new])
                print(len(raw))
        print("Data loaded")
        raw = raw[['content','troll']]
        x = raw['content'].apply(lambda s: read_words(s)).to_numpy()
        y = raw['troll'].to_numpy()
        print("Words parsed")
        print(len(x))
       
        if loadTokenizer == True:
            with open("tokenizer.pkl",'rb') as infile:
                tokenizer = pickle.load(infile)
                print("Loaded tokenizer")
        else:
            tokenizer = Tokenizer(num_words=MAX_NB_WORDS+1, oov_token="UNK")
            tokenizer.fit_on_texts(x)
            word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS}
            word_index[tokenizer.oov_token] = MAX_NB_WORDS + 1
            print('Found %s unique tokens' % len(word_index))
            with open('tokenizer.pkl', 'wb') as outfile:
                pickle.dump(tokenizer, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            
            # prepare embeddings
            print('Preparing embedding matrix')
            word2vec = gensim.load("glove-twitter-200")   # get pretrained glove model trained on 2Billion tweets

            embedding_matrix = np.zeros((MAX_NB_WORDS+1, EMBEDDING_DIM))
            for word, i in word_index.items():    
                if word in word2vec.vocab:    
                    embedding_matrix[i] = word2vec.word_vec(word)
            print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        # embed and pad
        x = tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)

        save_obj = {'x': x, 'y': y, 'embedding_matrix': embedding_matrix}
        with open(path+file+"_all.pkl",'wb') as outfile:
            pickle.dump(save_obj, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data dumped")

    else:
        with open(path+file+"_all.pkl",'rb') as infile:
            save_obj = pickle.load(infile)
        print("Data loaded")
        x = save_obj['x']
        y = save_obj['y']
        embedding_matrix = save_obj['embedding_matrix']

    return x, y, embedding_matrix


class KerasBatchGenerator(object):

    def __init__(self, x_data, y_data, batch_size):
        self.x_data = x_data
        self.y_data = y_data
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0

    def generate(self):
        x = np.zeros((self.batch_size, MAX_SEQUENCE_LENGTH))
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
    X, y, embedding_matrix = load_data('tweet_data_batch0', first)
    if run == 'train':
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, shuffle=True)
        batch_size = 200
        print("Begin train")
        train_data_generator = KerasBatchGenerator(X_train, y_train, batch_size)
        valid_data_generator = KerasBatchGenerator(X_valid, y_valid, batch_size)

        hidden_size = 500

        if loadOld:
            print("Load model")
            model = load_model(path + "true_model-10.hdf5")
        else:
            print("Create model")
            model = Sequential()
            model.add(Embedding(MAX_NB_WORDS+1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
            model.add(LSTM(hidden_size, return_sequences=True))
            model.add(LSTM(hidden_size, return_sequences=False))
            if use_dropout:
                model.add(Dropout(0.5))

            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            optimizer = Adam()
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        checkpointer = ModelCheckpoint(filepath=path + 'true_model-{epoch:02d}.hdf5', verbose=1)
        num_epochs = 20
        model.fit_generator(train_data_generator.generate(), 2000, num_epochs, validation_data=valid_data_generator.generate(), validation_steps=100, callbacks=[checkpointer])
        model.save(path + "true_final_model.hdf5")
    elif run == 'test':
        batch_size = 750
        model = load_model(path + "true_final_model.hdf5")
        data_generator = KerasBatchGenerator(X, y, batch_size)
        score = model.evaluate(data_generator.generate(), steps=2000, verbose=1)
        for i in range(len(score)):
             print("%s: %.4f" % (model.metrics_names[i], score[i]))
        y_true = [1 if point else 0 for point in y]
        y_pred=model.predict(X, verbose=1)
        con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred)
        sess = tf.Session()
        with sess.as_default():
            print(con_mat.eval())

