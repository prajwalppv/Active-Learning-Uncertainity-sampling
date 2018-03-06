import active as a
import passiveRandomLearners as r
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

def blinded_predictions(model,mode):
    '''
    This funcion makes predictions on the blinded dataset

    Inputs:
        1) model - trained active learner
        2) mode - type of dataset "EASY","MODERATE" or "DIFFICULT"
    
    Outputs:
        Creates a csv file with the predictions
    '''
    out_file = mode + "_BLINDED_PRED.csv"
    writer = csv.writer(open(out_file,"w"))
    file_name = mode + "_BLINDED.csv"
    df = pd.read_csv(file_name,header=None)
    ids = df.iloc[:,0].values
    df_X = df.drop(0,axis=1)
    X = df_X.values
    y = model.predict(X)
    for id,pred in zip(ids,y):
        writer.writerow([id,pred])

def active_random(m="easy",iterations = 5,base="gaussianNB"):
    '''
    This function creates the plots of loss curves of active,passive and random learners as a function of number of calls to the oracle.

    Inputs:
        1) m - type of dataset "EASY","MODERATE" or "DIFFICULT"
        2) iterations - number of rounds to run the models to average out losses/accuracies
        3) base - Required base leaner "gaussianNB" or "randomForest"
    
    Outputs:
        Creates 3 graphs which are the loss curves vs number of calls to the oracle on the given dataset
    '''
    print("Running on ",m, " dataset")
    #random_loss holds the average loss of random learner over the rounds
    random_loss = np.empty(10)
    for x1 in range(len(random_loss)):
        random_loss[x1] = 0
    
    #passive_loss holds the average loss of passive learner over the rounds
    passive_loss = np.empty(10)
    for x1 in range(len(passive_loss)):
        passive_loss[x1] = 0

    #active_loss holds the average loss of active learner over the rounds    
    active_loss = np.empty(2250)
    for x2 in range(len(active_loss)):
        active_loss[x2] = 0

    #If base learner is  randomForest, run only 1 round for active learner to save time    
    if(base == "randomForest"):
        nb_active_model = a.active_learner(m,base)
        active_loss = nb_active_model[1]
    for iter in range(iterations):
        print("Iteration ",iter+1, " out of ",iterations)
        if base!="randomForest":
            nb_active_model = a.active_learner(m,base)
            active_loss = np.add(active_loss,np.array(nb_active_model[1]))

        random_model = r.learner("random",m,base)
        passive_model = r.learner("passive",m,base)
        
        random_loss = np.add(random_loss , np.array(random_model[0][1]))
        passive_loss = np.add(passive_loss , np.array(passive_model[0][1]))

        active_x = nb_active_model[0]
        random_x = random_model[0][0]
        passive_x = passive_model[0][0]

    #Get average loss over the rounds for all models
    if base!="randomForest":
        active_loss = list(active_loss/iterations)
    random_loss = list(random_loss/iterations)
    passive_loss = list(passive_loss/iterations)

    #active_xprime and active_yprime are used to store the required (x,y) points to plot on the graph
    active_xprime = []
    active_yprime = []
    for i in range(0,len(active_x)):

        if(active_x[i]==251 or active_x[i]%250==0):
            active_xprime.append(active_x[i])
            active_yprime.append(active_loss[i])

    accu_active = 1 - active_loss[-1]
    accu_random = 1 - random_loss[-1]
    print("Accuracy of active model = ",accu_active)
    print("Accuracy of random learner = ",accu_random)

    print("Making prediction on blinded data")
    blinded_predictions(nb_active_model[3],m)
    print("Predictions Completed on blinded data set")

    #Plot loss curves vs Number of calls to the oracle
    plt.title("Loss of different learners on "+m+ " dataset")
    if base =="randomForest":
        a_label = "Active " + base + " (Query by committee)"
    else:
        a_label = "Active " + base + " Uncertainty Sampling" 

    green_patch = mpatches.Patch(color='green', label=a_label)
    red_patch = mpatches.Patch(color='red', label='Random Learner')
    black_patch = mpatches.Patch(color='black', label='Passive Learner')

    plt.xlabel("Number of calls to the oracle")
    plt.ylabel("Fraction of points misclassified")
    plt.legend(handles=[red_patch,green_patch,black_patch])#blue_patch])

    plt.plot(random_x,random_loss,"r--",active_xprime,active_yprime,"g",passive_x,passive_loss,"black")
    plt.savefig("Active_Random_"+m+"_"+base+".png")
    plt.clf()

#Run the program for gaussianNB on all 3 datasets
modes = ["EASY","MODERATE","DIFFICULT"]
for mo in modes:
    active_random(mo)