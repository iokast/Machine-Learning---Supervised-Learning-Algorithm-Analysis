# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 04:04:40 2018

@author: Brandon

Modified from Jonathan Tay code:
https://github.com/JonathanTay/CS-7641-assignment-1
"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knnC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
 

# Load Data       
seg = pd.read_hdf('datasets.hdf','segmentation')     
segX = seg.drop('classification',1).copy().values
segY = seg['classification'].copy().values
le = preprocessing.LabelEncoder()
segY = le.fit_transform(segY)


# split dataset into training (70%) and test (30%) sets    
seg_trgX, seg_tstX, seg_trgY, seg_tstY = ms.train_test_split(segX, segY, test_size=0.3, random_state=0,stratify=segY)     


## =============================================================================
## DECISION TREE
## =============================================================================


pipeSeg = Pipeline([('Scale',StandardScaler()),
                 ('DT',DecisionTreeClassifier(random_state=55))])


params_seg = {'DT__max_depth':np.arange(5,30,1),'DT__min_samples_leaf':np.arange(1,16,1),'DT__class_weight':['balanced']}

seg_clf = basicResults(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,params_seg,'DT','seg')  

seg_final_params = seg_clf.best_params_

pipeSeg.set_params(**seg_final_params)
makeTimingCurve(segX,segY,pipeSeg,'DT','seg')


## =============================================================================
## BOOST
## =============================================================================

pipeSeg = Pipeline([('Scale',StandardScaler()),
                 ('Boost',AdaBoostClassifier(DecisionTreeClassifier(), random_state=1))])

max_depth_seg = np.arange(1,21,1)

params_seg = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,100],'Boost__base_estimator__max_depth':max_depth_seg,
                 'Boost__base_estimator__class_weight':['balanced']}


seg_clf = basicResults(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,params_seg,'Boost','seg') 
 
seg_final_params = seg_clf.best_params_

pipeSeg.set_params(**seg_final_params)
makeTimingCurve(segX,segY,pipeSeg,'Boost','seg')      


## =============================================================================
## KNN
## =============================================================================

pipeSeg = Pipeline([('Scale',StandardScaler()),
                 ('KNN',knnC())])

params_seg= {'KNN__metric':['manhattan','euclidean'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
  
seg_clf = basicResults(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,params_seg,'KNN','seg') 

seg_final_params=seg_clf.best_params_

pipeSeg.set_params(**seg_final_params)
makeTimingCurve(segX,segY,pipeSeg,'KNN','seg')


## =============================================================================
## ANN
## =============================================================================

# create pipeline for each dataset
pipeSeg = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=500,early_stopping=True,random_state=55))])   

# set hyperparameter grid
d = segX.shape[1]
hiddens_seg = [(h,)*l for l in [1,2,3] for h in [d//2,int(d//1.5), d,int(d*1.5), d*2]]

learning_rate = np.logspace(-5, 1, 14)

params_seg = {'MLP__learning_rate_init':learning_rate,'MLP__hidden_layer_sizes':hiddens_seg }

# fit classifier to data using gridsearch and create learning curve data
seg_clf = basicResults(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,params_seg,'ANN','seg')        

# extract optimal parameters and create additional set of parameters with no regularization
seg_final_params =seg_clf.best_params_

# make timing curve using optimal parameters
pipeSeg.set_params(**seg_final_params)
pipeSeg.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(segX,segY,pipeSeg,'ANN','seg')

# generate learning curve data based on number of iterations with optimal parameters     
pipeSeg.set_params(**seg_final_params)
pipeSeg.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,{'MLP__max_iter':[2**x for x in range(13)]},'ANN','seg')       


## =============================================================================
## SVM  (Linear and RBF)       
##               
## =============================================================================

#Linear SVM
pipeSeg = Pipeline([('Scale',StandardScaler()),
                 ('SVM',LinearSVC())])

# define parameter grid
C_range = np.logspace(-2, 10, 13)

params_seg = {'SVM__C':C_range}
                                               
 # fit classifier to data using gridsearch and create learning curve data  
seg_clf = basicResults(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,params_seg,'SVM_LIN','seg')  
      
seg_final_params = seg_clf.best_params_

pipeSeg.set_params(**seg_final_params)                     
makeTimingCurve(segX,segY,pipeSeg,'SVM_LIN','seg')

pipeSeg.set_params(**seg_final_params)
iterationLC(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,{'SVM__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'SVM_LIN','seg') 


# RBF SVM
pipeSeg = Pipeline([('Scale',StandardScaler()),
                 ('SVM',SVC())])

# define parameter grid
C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)

#params_seg = {'SVM__n_iter':[int((1e6/N_seg)/.8)+1],'SVM__gamma':gamma_range,'SVM__C':C_range}
params_seg = {'SVM__C':C_range}

                                               
 # fit classifier to data using gridsearch and create learning curve data  
seg_clf = basicResults(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,params_seg,'SVM_RBF','seg')  
      
seg_final_params = seg_clf.best_params_

pipeSeg.set_params(**seg_final_params)                     
makeTimingCurve(segX,segY,pipeSeg,'SVM_RBF','seg')

pipeSeg.set_params(**seg_final_params)
iterationLC(pipeSeg,seg_trgX,seg_trgY,seg_tstX,seg_tstY,{'SVM__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'SVM_RBF','seg')  
