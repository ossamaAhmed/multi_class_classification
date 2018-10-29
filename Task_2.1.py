#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:23:10 2018

@author: yanni
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


#Input file names
X_data="X_train.csv"
y_data="y_train.csv"
X_fresh="X_test.csv"

#Import csv files as data frames and set all missing values to zero for now.
df0= pd.read_csv(X_data,header = 0)
df1=pd.read_csv(y_data,header = 0)
df2= pd.read_csv(X_fresh,header = 0)

#Extract, from the data frames, the 2D arrays of values
X_traindata=df0.values[:,-1000:]
y_traindata=df1.values[:,1]
X_fresh=df2.values[:,-1000:]

#Train Test split
X_train=X_traindata
y_train=y_traindata

#Pipeline
#pca=PCA();

#wclf=svm.SVC(C=0.975,kernel='rbf',class_weight='balanced',decision_function_shape='ovr')
#wclf=wclf.fit(X_train,y_train)

from sklearn.multioutput import ClassifierChain

intermediate=df1.y.replace({2:0});
y_train = np.c_[y_train,intermediate]
svcc = svm.SVC(C=0.875,kernel='rbf',class_weight='balanced',decision_function_shape='ovo',gamma='scale',probability=True)
cclf= ClassifierChain(base_estimator=svcc, order=[1,0], cv = 3)
cclf.fit(X_train,y_train)

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(cclf, X_train, y_train, cv=5,scoring = 'balanced_accuracy',verbose=3,n_jobs=-1)
#print(scores.mean())
#print(scores.std())

"""
from sklearn.model_selection import cross_val_predict
predicted=cross_val_predict(cclf,X_train,y_train,cv=5,n_jobs=-1,verbose=3)
chained_score=balanced_accuracy_score(y_train[:,0],predicted[:,0])
print(chained_score)
"""

"""
pipeline=Pipeline(steps=[('clf', wclf)])
pipeline=pipeline.fit(X_train, y_train)   

#Use pipeline.get_params().keys() to find how to assign hyperparameters
hyperparameters = {'clf__C':[0.9,0.95,1,1.5]}

grid = GridSearchCV(pipeline, hyperparameters, scoring = 'balanced_accuracy', cv=10,verbose=3,n_jobs=-1)
estimator=grid.fit(X_train,y_train).best_estimator_

#Print the mean cross validated test score of the best combination of parameters from GridSearchCV
print(grid.best_score_)

"""

#Output CSV file
forecast_set=np.matrix(cclf.predict(X_fresh)[:,0]).T
indexes=df2['id'].values;
dd=np.insert(forecast_set,0,indexes,axis=1)
df=pd.DataFrame(dd)
df.to_csv('forecast.csv',sep=',',float_format='%s',index=False, header=['id','y'])


    
