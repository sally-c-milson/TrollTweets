import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

path = "./data/"

def load_data(file):
	raw = pd.DataFrame()
	for i in range(12):		#currently using data compiled from first 50 csv files
		new = pd.read_csv(path+file+str(i+1)+".csv")
		raw = pd.concat([raw,new])
	raw = raw[['content','troll']]
	#these can be played with, currently set to ignore words in more than half or less than 100
	vectorizer = TfidfVectorizer(min_df=100,max_df=0.5) 
	c =  vectorizer.fit_transform(raw['content'])
	dictionary = vectorizer.get_feature_names()
	return c, raw['troll'], dictionary


X, y, dictionary = load_data("tweet_data_batch")
X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle = True)

model1 = LogisticRegression(max_iter=400,n_jobs=-1) #increased arbitariy to 400 since default iterations reach limit
'''
scores_clf_svc_cv1 = cross_val_score(model1,X,y,cv=5)
print("LogReg Accuracy: %0.2f (+/- %0.2f)" % (scores_clf_svc_cv1.mean(), scores_clf_svc_cv1.std() * 2))  # print accuracy
'''
model1.fit(X_train,y_train)
print("LogReg Accuracy:\n",model1.score(X_test,y_test))
predic1 = model1.predict(X_test)
print("LogReg matrix:",metrics.confusion_matrix(y_test,predic1))

model2 = Perceptron()
'''
scores_clf_svc_cv2 = cross_val_score(model2,X,y,cv=5)
print("Perceptron Accuracy: %0.2f (+/- %0.2f)" % (scores_clf_svc_cv2.mean(), scores_clf_svc_cv2.std() * 2))  # print accuracy
'''
model2.fit(X_train,y_train)
print("Perceptron Accuracy:\n",model2.score(X_test,y_test))
predic2 = model2.predict(X_test)
print("Perceptron matrix:", metrics.confusion_matrix(y_test,predic2))


'''
for 50 csv files, without cv
LogReg Accuracy:
 0.921791833442051
LogReg matrix: [[687272  25334]
 [ 55715 268003]]
Perceptron Accuracy:
 0.8803192823865895
Perceptron matrix: [[639105  73501]
 [ 50527 273191]]

for 16 csv files, without cv
LogReg Accuracy:
 0.9569325
LogReg matrix: [[ 65300  10190]
 [  7037 317473]]
Perceptron Accuracy:
 0.9411025
Perceptron matrix: [[ 62963  12527]
 [ 11032 313478]]

WITH 5 FOLD CV:
for 50 csv files, with cv
LogReg Accuracy: 0.82 (+/- 0.24)
Perceptron Accuracy: 0.80 (+/- 0.21)

for 16 csv files, with cv
LogReg Accuracy: 0.95 (+/- 0.02)
Perceptron Accuracy: 0.93 (+/- 0.03)
'''

