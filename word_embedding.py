import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import gensim.downloader as gensim
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics

def get_data():
    df = pd.read_csv('all_files.csv', encoding = "ISO-8859-1")
    df = df[df.index % 10 == 0] # grab every 10th row 
    return df


def preprocess(df):
    content_data= []
    for i in df.itertuples():
        text = i[3]
        text = text[2:-1]
        corpus = [word.lower() for word in text.split()]
        content_data.append(corpus) #content data is a list of a sentence for context
    y = df.get('troll')
    return content_data, y

def train(data):
    sg_model = Word2Vec(data, min_count=5,size= 100, window =5, sg = 1,workers = -1) #skip-gram algorithim
    cbow_model = Word2Vec(data, min_count=5,size= 100, window =5, sg = 0,workers = -1) #continuous bag of words algorithim

    return sg_model, cbow_model

def get_word_vec(dictionary, model):
    word_vec = pd.DataFrame()
    for word in dictionary:
        vector = model.wv[word]
        word_vec[word] = vector
    return word_vec


        
if __name__ == "__main__":
    df = get_data()
    content_data, y = preprocess(df)
    sg_model, sbow_model = train(content_data)   #train word2vec models
    glove_model = gensim.load("glove-twitter-25")   #get pretrained glove model trained on 2Billion tweets
    sg_dictionary = list(sg_model.wv.vocab)
    X = get_word_vec(sg_dictionary,sg_model)
    y= y.head(len(X.index))

#problems - for train_test_split, the X, y lenghts have to be the same but if we truncate y to X's lenght, it only contains True class    

X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle = True)

model1 = LogisticRegression(max_iter=400,n_jobs=-1) #increased arbitariy to 400 since default iterations reach limit
model1.fit(X_train,y_train)
print("LogReg Accuracy:\n",model1.score(X_test,y_test))
predic1 = model1.predict(X_test)
print("LogReg matrix:",metrics.confusion_matrix(y_test,predic1))
#######################################################
