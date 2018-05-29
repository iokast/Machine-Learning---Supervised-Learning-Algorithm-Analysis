# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:42:53 2018

@author: Brandon

Modified from Jonathan Tay code:
https://github.com/JonathanTay/CS-7641-assignment-1
"""

import pandas as pd

poker = pd.read_csv('./datasets/poker-hand-training-true.data', header=None)
poker.columns = ['s1','c1','s2','c2','s3','c3','s4','c4','s5','c5','hand'] 
poker.to_hdf('datasets.hdf','poker',complib='blosc',complevel=9)

poker = pd.read_csv('./datasets/poker-hand-testing.data', header=None)
poker.columns = ['s1','c1','s2','c2','s3','c3','s4','c4','s5','c5','hand']   
poker.to_hdf('datasets.hdf','poker_test',complib='blosc',complevel=9)

#covtype = pd.read_csv('./datasets/covtype.data', header=None)
#covtype.to_hdf('datasets.hdf','covtype',complib='blosc',complevel=9)



segmentation = pd.read_csv('./datasets/segmentation.csv',header=None)
#le = LabelEncoder()
#segmentation = segmentation.values
#segmentation[:,0] = le.fit_transform(segmentation[:,0])
#segmentation = pd.DataFrame(segmentation)
segmentation.columns = ['classification','REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2',
                        'VEDGE-MEAN','VEDGE-SD','HEDGE-MEAN','HEDGE-SD','INTENSITY-MEAN','RAWRED-MEAN','RAWBLUE-MEAN','RAWGREEN-MEAN',
                        'EXRED-MEAN','EXBLUE-MEAN','EXGREEN-MEAN','VALUE-MEAN','SATURATION-MEAN','HUE-MEAN']
segmentation.to_hdf('datasets.hdf','segmentation',complib='blosc',complevel=9)





# =============================================================================
# poker = pd.read_csv('./datasets/poker-hand-testing.data', header=None)
# poker.columns = ['s1','c1','s2','c2','s3','c3','s4','c4','s5','c5','hand']
# print(poker.describe(include='all'))
# pd.options.display.max_columns = 2000    
# poker.to_hdf('mushrooms.hdf','pokertest',complib='blosc',complevel=9)
# 
# =============================================================================

