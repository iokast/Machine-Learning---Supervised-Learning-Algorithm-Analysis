# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:48:00 2018

@author: Brandon

Modified from Jonathan Tay code:
https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd
from helpers import basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knnC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

 
# load training set      
poker = pd.read_hdf('datasets.hdf','poker') 
poker_trgX = poker.drop('hand',1).copy().values
poker_trgY = poker['hand'].copy().values

# Load test set
poker = pd.read_hdf('datasets.hdf','poker_test') 
poker_tstX = poker.drop('hand',1).copy().values
poker_tstY = poker['hand'].copy().values
     
# combine sets for use only in timing curve generation
pokerX = np.concatenate((poker_trgX, poker_tstX), axis=0)
pokerY = np.concatenate((poker_trgY, poker_tstY), axis=0)

# =============================================================================
# DECISION TREE
# =============================================================================
#
#min_samples_leaf = np.arange(1,11,1)
#depth_poker = np.arange(10,31,1)

pipePoker = Pipeline([('Scale',StandardScaler()),
                 ('DT',DecisionTreeClassifier(random_state=55))])

# exploratory trial runs showed depth optimal around 14 for poker and 22 for poker,
# gini performed best, min_samples_leaf was between 1 and 15.
params_poker = {'DT__max_depth':np.arange(10,31,1), 'DT__min_samples_leaf':np.arange(1,16,1), 'DT__class_weight':['balanced']}

poker_clf = basicResults(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'DT','poker')  

poker_final_params = poker_clf.best_params_

pipePoker.set_params(**poker_final_params)
makeTimingCurve(pokerX,pokerY,pipePoker,'DT','poker')
#
#
# =============================================================================
# BOOST
# =============================================================================

pipePoker = Pipeline([('Scale',StandardScaler()),
                  ('Boost',AdaBoostClassifier(DecisionTreeClassifier(), random_state=1))])

max_depth_poker = np.arange(1,15,1)
# min_samples_leaf_poker = np.arange(5,20,4)

params_poker = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,100],'Boost__base_estimator__max_depth':max_depth_poker, 
                'Boost__base_estimator__class_weight':['balanced']}


poker_clf = basicResults(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'Boost','poker')
 

poker_final_params = poker_clf.best_params_

pipePoker.set_params(**poker_final_params)
makeTimingCurve(pokerX,pokerY,pipePoker,'Boost','poker')  

# =============================================================================
# KNN
# =============================================================================

pipePoker = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])

params_poker= {'KNN__metric':['manhattan','euclidean'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
  
poker_clf = basicResults(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'KNN','poker') 

poker_final_params=poker_clf.best_params_

pipePoker.set_params(**poker_final_params)
makeTimingCurve(pokerX,pokerY,pipePoker,'KNN','poker')


# =============================================================================
# ANN
# =============================================================================

# create pipeline for each dataset
# max iter selected via early testing
pipePoker = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=500,early_stopping=True,random_state=55))])   

# set hyperparameter grid
d = pokerX.shape[1]
hiddens_poker = [(h,)*l for l in [1,2,3] for h in [5,10,15,20,25,30]]

learning_rate = np.logspace(-5, 1, 14)

params_poker = {'MLP__learning_rate_init':learning_rate,'MLP__hidden_layer_sizes':hiddens_poker }

# fit classifier to data using gridsearch and create learning curve data
poker_clf = basicResults(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'ANN','poker')        

# extract optimal parameters and create additional set of parameters
poker_final_params =poker_clf.best_params_

# make timing curve using best parameters
pipePoker.set_params(**poker_final_params)
pipePoker.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(pokerX,pokerY,pipePoker,'ANN','poker')

# generate learning curve data based on number of iterations with best parameters     
pipePoker.set_params(**poker_final_params)
pipePoker.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'MLP__max_iter':[2**x for x in range(13)]},'ANN','poker')       


## =============================================================================
## SVM         
##               
## =============================================================================


# Linear SVM
pipePoker = Pipeline([('Scale',StandardScaler()),
                 ('SVM',LinearSVC())])

# define parameter grid
C_range = np.logspace(-2, 10, 13)

params_poker = {'SVM__C':C_range}

 # fit classifier to data using gridsearch and create learning curve data  
poker_clf = basicResults(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'SVM_LIN','poker')  
      
poker_final_params = poker_clf.best_params_

pipePoker.set_params(**poker_final_params)                     
makeTimingCurve(pokerX,pokerY,pipePoker,'SVM_LIN','poker')

pipePoker.set_params(**poker_final_params)
iterationLC(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'SVM__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'SVM_LIN','poker')  

# RBF SVM
pipePoker = Pipeline([('Scale',StandardScaler()),
                 ('SVM',SVC())])

# define parameter grid
C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)

#params_poker = {'SVM__n_iter':[int((1e6/N_poker)/.8)+1],'SVM__gamma':gamma_range,'SVM__C':C_range}
params_poker = {'SVM__C':C_range,'SVM__max_iter':[3000]}

                                               
 # fit classifier to data using gridsearch and create learning curve data  
poker_clf = basicResults(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,params_poker,'SVM_RBF','poker')  
      
poker_final_params = poker_clf.best_params_

pipePoker.set_params(**poker_final_params)                     
makeTimingCurve(pokerX,pokerY,pipePoker,'SVM_RBF','poker')

pipePoker.set_params(**poker_final_params)
iterationLC(pipePoker,poker_trgX,poker_trgY,poker_tstX,poker_tstY,{'SVM__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'SVM_RBF','poker')  