import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.patches as mpatches
from passiveRandomLearners import compute_loss


def calc_entropy(row):
    '''
    This function caluculates the entropy for each data point given the probabiities that the point is labelled as a certain class. The equation for calculating entropy is present in the report.

    Inputs:
        1)row : A 1-D array containing the probabilities that a point can be labelled as a class

    Outputs:
        1)entropy : entropy of the data point
    '''
    entropy = 0.0
    for r in row:
        if r!=0:
            entropy+= r * np.log(r)
    return -entropy

def active_learner(mode,base="gaussianNB"):
    '''
    Contains the implementation of the active learning algorithm based on Uncertainty sampling if the base is Gaussian Naive Bayes or Query-by-committee if the base is Random Forest Classifier

    Inputs:
        1) mode - Type of dataset i.e "EASY","MODERATE" or "DIFFICULT"
        2) base - Base learner we wish to use - can be "gaussianNB" or "randomForest"

    Ouputs:
        [x,loss,accuracy,model] where
        x - values to plot along the X-axis (basically stores the number of calls to the oracle at every iteration)
        loss - list of loss of model at each iteration
        accuracy - Accuracy of the model at each iteration
        rf_model - Trained active learning model
    '''
    print("----------------- Active ",base, ": ", mode,"-----------------")
    train_file = mode+ "_TRAIN.csv"
    test_file = mode + "_TEST.csv"

    #Sample 250 random points to initially train the active learning model
    initial_data_size= 249
    train_data = pd.read_csv(train_file)
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    initial_train_data = train_data.loc[0:initial_data_size,:]
    
    #Remove already sampled points
    train_data.drop(train_data.index[[x for x in range(0,initial_data_size)]],inplace=True)
    train_data.reset_index(inplace=True)
    train_data.drop("index",axis=1,inplace=True)
    
    test_data = pd.read_csv(test_file)

    #Depending on the dataset, select columns which are input features and use the last column as the lables
    if mode != "DIFFICULT":
        test_features = test_data.iloc[:,:26].values    
        test_labels = test_data.iloc[:,-1].values

        ini_train_features = initial_train_data.iloc[:,0:26].values
        ini_train_labels = initial_train_data.iloc[:,-1].values

        train_features = train_data.iloc[:,:26].values
        train_labels = train_data.iloc[:,-1].values

    else:
        test_features = test_data.iloc[:,:52].values
        test_labels = test_data.iloc[:,-1].values

        ini_train_features = initial_train_data.iloc[:,:52].values
        ini_train_labels = initial_train_data.iloc[:,-1].values

        train_features = train_data.iloc[:,:52].values
        train_labels = train_data.iloc[:,-1].values

        #Below commented code is the code to perform PCA on the data before training the model. I have not used PCA and it loses a lot of information and thus produces bad accuracy on test data.
        '''
        temp_data = np.vstack((ini_train_features,train_features))
        print("pca beginning")
        pca = PCA(n_components=26)
        temp_data = pca.fit_transform(temp_data)
        ini_train_features = temp_data[:initial_data_size+1,:]
        print(ini_train_features.shape)
        train_features = temp_data[initial_data_size+1:,:]
        test_features = pca.transform(test_features)
        '''
    
    #Initialize active learner based on given parameters
    if base=="randomForest":
        
        rf = RandomForestClassifier(n_estimators=100,criterion="gini")
        rf_model = rf.fit(ini_train_features,ini_train_labels)
    else:
       
        rf = GaussianNB()
        rf_model = rf.fit(ini_train_features,ini_train_labels)

    #Budget is the given stipulated budget of 2500
    budget = 2500 - initial_data_size

    loss = []       # Store loss at each iterations
    accuracy = []   # Store accuracy of model at each iteration
    x = []          # x stores the values we need to plot on the X-axis

    #Begin Active Learning Algorithm
    for i in range(1,budget):
        if i%500==0 or i==budget-1:
            print("Active learner -> Iteration ",i ," out of ", budget-1 )
        
        #Get probability of a belonging to each label for every point
        ypred = rf_model.predict_proba(train_features)
        #Calculate entropy of all points
        entropy_matrix = np.apply_along_axis(calc_entropy,1,ypred)
        #Choose point with maximum entropy
        max_ent = np.argmax(entropy_matrix)

        selected_data_point = train_features[max_ent,:]
        true_label = train_labels[max_ent]

        #Remove chosen point from dataset
        train_features = np.delete(train_features,[max_ent],axis = 0)
        train_labels = np.delete(train_labels,[max_ent])

        #Add chosen point to labelled dataset
        ini_train_features = np.vstack((ini_train_features,selected_data_point))
        ini_train_labels =  np.hstack((ini_train_labels,true_label))

        x.append(i+250)
        l = compute_loss(rf_model,test_features,test_labels)
        loss.append(l)

        #acc = rf_model.score(test_features,test_labels)
        acc = 1 - l
        accuracy.append(acc)
        rf_model = rf.fit(ini_train_features,ini_train_labels)

    return ([x,loss,accuracy,rf_model])