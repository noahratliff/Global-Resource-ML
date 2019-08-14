from sklearn.neural_network import MLPClassifier
from subprocess import call
from sklearn.tree import export_graphviz
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import pickle
import statsmodels.api as sm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC  # "Support vector classifier"
import eli5  # for purmutation importance
from eli5.sklearn import PermutationImportance
from sklearn.ensemble.bagging import BaggingClassifier

#Import all necesary libararies, used for modeling and data processing

df = pd.read_csv("NFA 2018.csv")

#Read NFA Data into pandas dataframe

df = df.loc[df['record'] == 'EFProdTotGHA']

#Isolate total data by country

print(df.columns)

df = df.drop(['ISO alpha-3 code', 'UN_region', 'UN_subregion'], 1)

#Remove features that detract from model accuracy

print(df.head())
df.dropna(inplace=True)

#Drop blank values

plt.style.use('seaborn')

X = np.array(df.drop('population', 1))

#Set features to all items except population

y = np.array(df['population'])

#Set label to population

feature_names = [i for i in df.drop('population', 1).columns]

target_names = ['population']

np.random.seed(0)
msk = np.random.rand(len(df)) < 0.8

trainX = X[msk]
testX = X[~msk]
trainY = y[msk]
testY = y[~msk]
print(testY)

#Section data into testing and training

def warn(*args, **kwargs):
    pass


warnings.warn = warn

rf_class = RandomForestClassifier(n_estimators=400)
bc = BaggingClassifier()
sgd = SGDClassifier()
bnb = BernoulliNB()

#Assign models to variables

def run(model, model_name='this model', trainX=trainX, trainY=trainY, testX=testX, testY=testY):
    # print(cross_val_score(model, trainX, trainY, scoring='accuracy', cv=10))
    accuracy = cross_val_score(model, trainX, trainY,
                               scoring='accuracy', cv=10).mean() * 100
    model.fit(trainX, trainY)
    testAccuracy = model.score(testX, testY)
    print("Training accuracy of "+model_name+" is: ", accuracy)
    print("Testing accuracy of "+model_name+" is: ", testAccuracy*100)
    print('\n')

#Create function for running and assessing the accuracy of a model on a given dataset

run(sgd,'sgd')
run(bc,'bc')
run(rf_class, 'forest')
run(bnb,'bnb')
model = bc

#Run 4 different models for accuracy

perm = PermutationImportance(model, random_state=1).fit(testX, testY)
eli5.show_weights(perm, feature_names=feature_names)

#Display the positive and negative correlations between different variables for quantitative assessment

results = model.predict(testX)

plt.plot(trainY,'red')
plt.plot(testY, color = 'green')
print(testX)
print(testY)
print('\n\n\n\n\n')
plt.plot(results,color = 'blue')

#Graph the training and testing data for qualitative assessment

