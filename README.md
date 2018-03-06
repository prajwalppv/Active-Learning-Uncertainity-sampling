**************************************************************************************************
Image Classification using Active Learning - Uncertaininty sampling

Author: Prajwal Prakash Vasisht

Andrew ID: prajwalp@andrew.cmu.edu

Last modified date : 12/10/2017
**************************************************************************************************

Python Dependencies:-
The code has been written to work in Python v3.6.x
Following python libraries are required to be installed in order to run the program.
    1) scikit-learn
    2) pandas
    3) matplotlib
    4) numpy
    5) scipy
    6) seaborn
    7) csv
    8) random

HOW TO RUN :
    
    1) In order to generate loss curves for active, random and passive learners on all 3 datasets and make predictions on blinded datasets -> python3 plot_active.py 
    It will generate 6 plots, 3 each for GaussianNB active learner and randomForest active learner

    2) In order to generate plots of passive learers -> python3 plot_passive.py
    It will generate 6 plots, 3 loss curve graphs and 3 accuracy curve graphs for the comparison of base learners. 

Code descriptions:-
    This section outlays the function of each python file in the "Code" folder. More details are present as comments in the code file.
        
    1) active.py - active.py contains the code for the active learner. It returns losses,accuracies and the trained active learner model to the calling function. 

    2) passiveRandomLearners.py - This file contains the code for passive and random learners.

    3) plot_active.py - This file is the driver program that compares the passive, random and Active learners and produces graphs of loss curves as a function of number of calls to the oracle and accuracy values for all 3 datasets for both gaussian naive bayes and random forest classifiers.

    4) plot_passive.py - This file is the driver program that compares multiple base (passive) learners and plots graphs of the losses and accuracies as a function of number of data points seen(used).

Folder contents:-
    
    1) The "Code" folder contains all the python code and required data files needed to run the code.

    2) New images generated by the program will be present in the same folder as the program is located in. The "Images" folder has a backup of all images used in the report.

    3) The "Predictions" folder contians the predictions of the trained active learning model on all 3 blinded data sets.

    4)The "Report" folder contains the final project report and other miscellaneous files.

    5) the logbook.txt is just a log keeping file for the various parameter combinations I tested out during the course of the project. It is not an exhaustive list of all the models I have tried.

Note: In case run fails, please clear __pycache__ and try again
