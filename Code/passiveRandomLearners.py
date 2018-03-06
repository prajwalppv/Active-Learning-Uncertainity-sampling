import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.patches as mpatches
import random

def compute_loss(model,testData,labels):
    '''
    This funtion computes the loss of the model at a given iteration. Loss here has been defined as the fraction of misclassified points by the model.

    Inputs:
        1) model - Trained model
        2) testData - test input features
        3) labels - True labels for the given test data

    Outputs:
        1) loss - fraction of wrongly classified points
    '''
    predicted_labels = model.predict(testData)
    count = 0
    for predicted,true in zip(predicted_labels,labels):
        if predicted != true:
            count+=1
    loss = count/len(labels)
    if(loss>1):
        loss = 1
    return loss


def learner(type,mode,base="gaussianNB"):
    '''
    This funtions trains a passive or random learner based on the given input parameters.

    Inputs:
        1) type - "passive" learner or "random" learner
        2) mode - Type of dataset "EASY", "MODERATE" or "DIFFICULT"
        3) base - Base learner to use - "gaussianNB" or "randomForest" or "svm"

    Ouputs:
        [losses,accuracies] where,
        losses - holds the loss of the model at each iteration
        accuracies - holds the accuracies of the model at each iteration
    '''

    sns.set()
    losses = []
    accuracies = []

    svm_loss = []
    rf_loss = []
    nb_loss = []

    svm_accuracy = []
    rf_accuracy = []
    nb_accuracy = []

    x = []

    train_file = mode+ "_TRAIN.csv"
    test_file = mode + "_TEST.csv"
    blinded_file = mode + "_BLINDED.csv"

    train_data = pd.read_csv(train_file)

    #If type is passive, Initialize a random starting point and choose the next 2500 consecutive points
    if type == "passive":
        start_index = random.randint(0,1500)
        end_index = start_index + 2500
        train_data = train_data.iloc[start_index:end_index,:]

    #If type is random, uniformly sample 2500 points from the dataset
    if type == "random":
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        train_data = train_data.iloc[:2500,:]

    test_data = pd.read_csv(test_file)

    #Extract input features and labels based on type of dataset
    if mode != "DIFFICULT":
        test_features = test_data.iloc[:,:26]
        test_labels = test_data.iloc[:,-1]
    else:
        test_features = test_data.iloc[:,:52]
        test_labels = test_data.iloc[:,-1]

    print("-----------------",type," ",base,": ", mode,"-----------------")

    minb = 250
    #batch_size defines how many input samples we process at a time
    batch_size = 250
    iter = int(2500/batch_size)
    for i in range(iter):
        if mode !="DIFFICULT":
            train_features = train_data.iloc[:batch_size,:26]
            train_labels = train_data.iloc[:batch_size,-1]

        else:
            train_features = train_data.iloc[:batch_size,:52]
            train_labels = train_data.iloc[:batch_size,-1]

        print(type , " learner ----> Iteration ",i+1 ," out of ", iter )

        #Train appropriate model based on "base" parameter
        if base == "svm":
        #Using multi-class SVM
            svm = LinearSVC(multi_class='ovr')
            
            svm_model = svm.fit(train_features,train_labels)
            
            svm_score = svm_model.score(test_features,test_labels)
            svm_l = compute_loss(svm_model,test_features,test_labels)
            svm_accuracy.append(svm_score)
            svm_loss.append(svm_l)

        elif base == "randomForest":
        #Using decision trees
            rf = RandomForestClassifier(n_estimators=100,criterion="gini")
            
            rf_model = rf.fit(train_features,train_labels)
            rf_l = compute_loss(rf_model,test_features,test_labels)
            rf_score = 1 - rf_l
            rf_accuracy.append(rf_score)
            rf_loss.append(rf_l)

        else:
            #Using Gaussian Naive Bayes classifier
            nb = GaussianNB()
        
            nb_model = nb.fit(train_features,train_labels)
            nb_l = compute_loss(nb_model,test_features,test_labels)
            nb_score = 1 - nb_l
            nb_accuracy.append(nb_score)
            nb_loss.append(nb_l)
       
        x.append(batch_size)
        batch_size += minb

    losses.append(x)
    accuracies.append(x)

    if(base == "svm"):    
        losses.append(svm_loss)
        accuracies.append(svm_accuracy)

    if base == "randomForest":
        losses.append(rf_loss)
        accuracies.append(rf_accuracy)
    else:    
        losses.append(nb_loss)
        accuracies.append(nb_accuracy)
    
    return ([losses,accuracies])
