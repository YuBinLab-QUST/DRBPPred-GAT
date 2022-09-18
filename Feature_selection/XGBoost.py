# -*- coding: utf-8 -*-
"""
Created on Thu May  7 20:33:44 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.preprocessing import scale,StandardScaler
#from RFE_feature_selection import SVM_RFE_selection,LOG_RFE_selection,XGB_selection,LGB_selection


#shu1 = pd.read_csv("N_zyb.csv",header=None)
#shu2 = pd.read_csv("N_fyb.csv",header=None)
#shu1 = np.array(shu1)
#shu2 = np.array(shu2)
#shu = np.concatenate((shu1,shu2),axis=0)
#
#[row1,column1]=np.shape(shu1)
#[row2,column2]=np.shape(shu2)
#label_P = np.ones(int(row1))
#label_N = np.zeros(int(row2))
#label = np.hstack((label_P,label_N))
data_=pd.read_csv(r'ALL.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(3846),1))#Value can be changed
label2=np.zeros((int(4175),1))
label=np.append(label1,label2)
shu=scale(data)
#test_network = pd.read_csv(r'N_ShiHan_network_ceshiji.csv')
#test_shu=np.array(test_network)
#test_shu_scale=scale(test_shu)

#shu=np.array(shu)
#label=np.array(label)
#shu = scale(shu)

xgb_model=xgb.XGBClassifier()#n_estimators=1000,max_depth=6
xgbresult1=xgb_model.fit(shu,label.ravel())
feature_importance=xgbresult1.feature_importances_
feature_number=-feature_importance
H1=np.argsort(feature_number)
mask=H1[:300]

train_data=shu[:,mask]
#test_data=test_shu_scale[:,mask]

X = train_data
#y = test_data
data_csv=pd.DataFrame(data=X)
data_csv.to_csv('ALL_XGB.csv')
#data_csv=pd.DataFrame(data=y)
#data_csv.to_csv('A_NR_network_SY_LGB_400.csv')
