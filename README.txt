CS 7641 Spring 2018 Assignment 1
Brandon Wetzel

NOTE: all code in this lab including this readme file is modified from code provide by Jonathan Tay:
https://github.com/JonathanTay/CS-7641-assignment-1

This file describes the structure of this assignment submission. 
The assignment code is written in Python 3.6.3. Library dependencies are: 
numpy: 1.13.3
pandas: 0.20.3
sklearn: 0.19.1

Other libraries used are part of the Python standard library. 

The main folder contains the following:
1... 'datasets' folder -> this folder contains three files that comprise the 2 datasets used in this asignment
-'poker-hand-testing.data' -> dataset 1 testing data from http://archive.ics.uci.edu/ml/
-'poker-hand-training-true.data' -> dataset 1 training data from http://archive.ics.uci.edu/ml/
-'segmentation.csv' -> dataset 2 downloaded from http://archive.ics.uci.edu/ml/ and compiled into a .csv document
2... 'datasets.hdf' -> A pre-processed/cleaned up copy of the datasets. This file is created by "parse data.py"
3... 'parse data.py' -> This python script pre-processes the original UCI ML repo files into a cleaner form for the experiments
4... 'bwetzel6-analysis.pdf' -> The analysis for this assignment.
5... 'helpers.py' -> A collection of helper functions used for this assignment
6... 'full_poker.py' -> Code for all algorithms using the Segmentation dataset
7... 'full_seg.py' -> Code for all algorithms using the Segmentation dataset
8... 'README.txt' -> This file
9... 'output' folder -> This folder contains the experimental results. 
-Here, I use DT/ANN/BT/KNN/SVM_LIN/SVM_RBF to refer to decision trees, artificial neural networks, boosted trees, K-nearest neighbours, linear and RBF kernel SVMs respectively. 
-The datasets are poker/seg referring to the two datasets used (the UCI Poker Hands dataset and the UCI Segmentation dataset)
-There are 61 files in this folder. They come the following types:
-a. '<Algorithm>_<dataset>_reg.csv' -> The validation curve tests for <algorithm> on <dataset>
-b. '<Algorithn>_<dataset>_LC_train.scv' -> Table of # of examples vs. CV training accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
-c. '<Algorithn>_<dataset>_LC_test.csv' -> Table of # of examples vs. CV testing accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
-d. '<Algorithm>_<dataset>_timing.csv' -> Table of fraction of training set vs. training and evaluation times. If the fulll training set is of size T and a fraction f are used for training, then the evaluation set is of size (T-fT)= (1-f)T
-e. 'ITER_base_<Algorithm>_<dataset>.csv' -> Table of results for learning curves based on number of iterations/epochs.
-f. 'ITERtestSET_<Algorithm>_<dataset>.csv' -> Table showing training and test set accuracy as number of iterations/epochs is varied. NOT USED in report.
-g. 'test results.csv' -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set.
10... 'excel masters' folder -> This folder contains consolidated data from output .csv's and was used to construc plots used in analysis