# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2  2020

@author: Ethan Bosworth

A script to classify the data preprocessed by the previous scripts
it was requested in the brief that a logistic regression was to be used
"""
#%% import modules
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np




#%% import data

data_train = pd.read_csv("../Case/Refined/train.csv")
data_test = pd.read_csv("../Case/Refined/test.csv")

#create a y from the target and an x from everything else
y = data_train["target"]
X = data_train.drop("target",axis = 1)


#%% Training a test model
#split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
#%%% create unchanged model


#create a classifier
LR = LogisticRegression(random_state = 1)
LR.fit(X_train,y_train) # fit the classifier
#create a prediction
y_pred = LR.predict(X_test)
y_pred_proba = LR.predict_proba(X_test)
#measure the accuracy of prediction and the confusion matrix
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
#the accuracy returns a good 0.812  but the confusion matrix shows 
#a high True Positive rate and a low true negative rate

#printing the counts of y shows it is because of heavily imbalanced classes
print(y.value_counts())

#%%% balancing the classes

#%%%%

#create a function that will resample the data based on different strategies
def resample(X,y,over_n,under_n):
    over = SMOTE(random_state = 1,sampling_strategy=over_n) # create a smote random oversampler
    under = RandomUnderSampler(random_state = 1,sampling_strategy=under_n) # create a random undersampler
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps) # create a pipeline to both oversample and undersample
    X_train_sampled, y_train_sampled = pipeline.fit_resample(X, y) # resample the data
    return X_train_sampled,y_train_sampled



#%%%% attempt with balanced classes

#create a function that we provide an i and j for the resampling and will return a dataframe
#containing accuracy and TP and TN rates
def testing(i,j):
    #run the function to resample based on i and j
    X_train_sampled,y_train_sampled = resample(X_train,y_train,i,j)
    
    #create a classifier
    LR_b = LogisticRegression(random_state = 1,max_iter=100000)
    LR_b.fit(X_train_sampled,y_train_sampled) # fit the classifier
    #create a prediction
    y_pred = LR_b.predict(X_test)
    return y_pred,y_pred_proba

#set up the ranges of i and j to run over
i_params = np.arange(0.3,0.8,0.1)
j_params = np.arange(0.4,1,0.1)
#create a dataframe of all possible combinations
params = pd.DataFrame(np.array(np.meshgrid(i_params,j_params)).T.reshape(-1,2))
params.columns = ["i","j"]
#create columns ready to hold the informaton
params["score"] = params["i"]*0
params["TP"] = params["i"]*0
params["TN"] = params["i"]*0

for i in params.index: # run over the size of the dataframe
    if params["i"][i] <= params["j"][i]: # only do something if i is bigger than j to avoid an error
        y_pred,y_pred_proba = testing(params["i"][i],params["j"][i]) # run the function from before to get the predictons
        if accuracy_score(y_test, y_pred) > 0.2: # an accuracy below 0.2 means it predicted everything the same and should be avoided
            params["score"][i] = accuracy_score(y_test, y_pred)  # put in the dataframe the score of the prediction
            CM = pd.DataFrame(confusion_matrix(y_test, y_pred)) # create a dataframe of the confusion matrix
            params["TP"][i] = CM[0][0] # take from the confusion matrix the TP and FN numbers and give to the params dataframe
            params["TN"][i] = CM[1][1]
params = params[params["score"] != 0] # discard all those where an error would occur or accuracy was too low

#%%%% best parameter test
y_pred,y_pred_proba = testing(0.3,0.9) # take the best parameter from the params dataframe
print(print(confusion_matrix(y_test, y_pred))) # show the confusion matrix in full

#%% Creating the classifier and predicting
#now the classes have been balanced I want to create the logistic classifier

#take the best resampled full data
X_sampled,y_sampled = resample(X,y,0.5,0.5)
print(y_sampled.value_counts())

# 0.3 0.9 predicts all as "1" and 0.5 0.5 gives more close results. needs testing

LR = LogisticRegression(random_state = 1)
LR.fit(X_sampled,y_sampled) # fit the classifier


#create a prediction
y_pred = LR.predict(data_test)
y_pred_proba = LR.predict_proba(data_test)
#measure the accuracy of prediction and the confusion matrix

results = pd.concat([pd.DataFrame(y_pred),pd.DataFrame(y_pred_proba[:,1])],axis = 1)
print(y_pred.sum())
results.columns = ["probability","target"]

#%%%  output the data



results.to_csv("../Case/Output/results.csv",index = False)