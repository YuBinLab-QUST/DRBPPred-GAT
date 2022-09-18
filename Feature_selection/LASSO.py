import scipy.io as sio
import numpy as np
import pandas as pd
#import itertools
#import matplotlib.pyplot as plt
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import SelectFromModel
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import StratifiedKFold
#import utils.tools as utils
#from dimensional_reduction import mutual_mutual
from sklearn.preprocessing import scale,StandardScaler
#from keras.layers import Dense, merge,Input,Dropout
#from keras.models import Model
from dimensional_reduction1 import Light_lasso

data_=pd.read_csv(r'ALL.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((3846,1))#Value can be changed
label2=np.zeros((4175,1))
label=np.append(label1,label2)
shu=scale(data)
X=shu
y=label
#ata_2,importance=Light_lasso(X,y.T.ravel(),0.05)
data_2,importance=Light_lasso(X,y,0.02)
shu=data_2 
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('ALL_Lasso.csv')
data_csv = pd.DataFrame(data=importance)
data_csv.to_csv('ALL_importance.csv')
