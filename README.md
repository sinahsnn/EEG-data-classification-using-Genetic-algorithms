# EEG-data-classification-using-Genetic-algorithms
In this repository, features were extracted from EEG data and best features were selected for classifying the signals using Genetic Algorithms.
in phase 1: 
- the features were extracted from the channels. 
- using fisher criterion the best features were chosen. 
- using 5-fold cross validation a MLP network was implemented to classify the data.
  -- * for better classification the hyperparameters such as number of layers, number of neurons, and activation functions were found. 
- the above step was done for implementing a RBF network whose hyperparameters such as number of neurons and their spreads were optimized.
