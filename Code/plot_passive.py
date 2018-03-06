import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.patches as mpatches
import passiveRandomLearners as r 


def plot_results(mode):
    '''
    Plots the graphs for loss and accuracy for svm, gaussianNB and randomForest passive/base learners

    Inputs:
        1) mode - type of dataset "EASY","MODERATE" or "DIFFICULT"
    
    Outputs:
        Produces 6 plots, 3 plots of loss curves and 3 plots of accuracy curves of the model through the iterations
    '''
    sns.set()

    svm = r.learner("passive",mode,base="svm")
    x = np.array(svm[0][0])

    rounds = 5

    svm_loss = []
    randomf_loss = []
    nbc_loss = []

    svm_accuracy = []
    randomf_accuracy = []
    nbc_accuracy = []

    average_rf_loss = np.zeros_like(x)
    average_nb_loss = np.zeros_like(x)
    average_svm_loss = np.zeros_like(x)
    
    average_rf_accuracy = np.zeros_like(x)
    average_nb_accuracy = np.zeros_like(x)
    average_svm_accuracy = np.zeros_like(x)

    for j in range(rounds):
        
        svm = r.learner("passive",mode,base="svm")
        randomf  = r.learner("passive",mode,base="randomForest")
        nbc = r.learner("passive",mode)

        svm_loss = svm[0][1]
        randomf_loss = randomf[0][1]
        nbc_loss = nbc[0][1]

        svm_accuracy = svm[1][1]
        randomf_accuracy = randomf[1][1]
        nbc_accuracy = nbc[1][1]

        average_rf_loss = np.add(average_rf_loss,np.array(randomf_loss))
        average_nb_loss = np.add(average_nb_loss,np.array(nbc_loss))
        average_svm_loss = np.add(average_svm_loss,np.array(svm_loss))

        average_rf_accuracy = np.add(average_rf_accuracy,np.array(randomf_accuracy))
        average_nb_accuracy = np.add(average_nb_accuracy,np.array(nbc_accuracy))
        average_svm_accuracy = np.add(average_svm_accuracy,np.array(svm_accuracy))

    average_rf_loss = average_rf_loss/rounds
    average_nb_loss = average_nb_loss/rounds
    average_svm_loss = average_svm_loss/rounds

    average_rf_accuracy = average_rf_accuracy/rounds
    average_nb_accuracy = average_nb_accuracy/rounds
    average_svm_accuracy = average_svm_loss/rounds

    #Plot passive learners for all datasets
    plt.title("Loss of base learners on "+mode+" dataset")
    red_patch = mpatches.Patch(color='red', label='Linear SVM')
    green_patch = mpatches.Patch(color='blue', label='Random Forest')
    blue_patch = mpatches.Patch(color='green', label='Gaussian Naive Bayes')
    plt.xlabel("Number of samples")
    plt.ylabel("Fraction of points misclassified")
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    plt.plot(x,svm_loss,"r",x,randomf_loss,"b",x,nbc_loss,"g")
    plt.savefig("Loss_"+mode+".png")
    plt.clf()

    plt.title("Accuracy of base learners on "+mode+" dataset")
    red_patch = mpatches.Patch(color='red', label='Linear SVM')
    green_patch = mpatches.Patch(color='blue', label='Random Forest')
    blue_patch = mpatches.Patch(color='green', label='Gaussian Naive Bayes')
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")
    plt.legend(handles=[red_patch,blue_patch,green_patch])
    plt.plot(x,svm_accuracy,"r",x,randomf_accuracy,"b",x,nbc_accuracy,"g")
    plt.savefig("Accuracy_"+mode+".png")
    plt.clf()

if __name__ == "__main__":
    modes = ["EASY","MODERATE","DIFFICULT"]
    for m in modes:
        plot_results(m)