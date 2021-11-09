import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import pipeline
from sklearn import compose
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_selection import SelectPercentile,chi2
import matplotlib.pyplot as plt
import numpy as np
import joblib


df = pd.read_csv('coronary_prediction.csv')
#df.isnull().sum()
cols = df.columns
cols_with_nan = [col for col in cols if df[col].isnull().any()]
features = df.drop('TenYearCHD',axis = 1)
target = df.TenYearCHD
x_train,x_test,y_train,y_test = model_selection.train_test_split(features,target,train_size = 0.7,random_state = 1)
#pipe = pipeline.make_pipeline(SimpleImputer(strategy='median'),SelectPercentile(chi2,percentile = 70),ensemble.AdaBoostClassifier())
#model_selection.cross_val_score(pipe,x_train,y_train,scoring='accuracy',cv = 10).mean()
params = {'adaboostclassifier__n_estimators':[10, 50, 100, 500, 1000, 5000],'adaboostclassifier__learning_rate':[0.2,0.3,0.4,0.5],'adaboostclassifier__algorithm':['SAMME.R','SAMME']}
#hypertuned = model_selection.GridSearchCV(pipe,params)
#hypertuned = model_selection.GridSearchCV(pipe,params,cv = 5,scoring = 'accuracy')
#hypertuned.fit(x_train,y_train)
#hypertuned.best_score_,hypertuned.best_params_,hypertuned.best_estimator_
final_pipe = pipeline.make_pipeline(SimpleImputer(strategy='median'),SelectPercentile(chi2,percentile = 70),ensemble.AdaBoostClassifier(algorithm='SAMME',learning_rate=0.3,n_estimators = 1000,random_state = None,base_estimator=None))
final_pipe.fit(x_train,y_train)
