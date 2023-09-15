import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.linear_model import ElasticNet,ElasticNetCV
#from dimensional_reduction import elasticNet


##using elasticNet to reduce the dimension
def elasticNet(data,label,alpha =np.array([0.05])):
    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.2).fit(data,label)
    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.2)
    enet.fit(data,label)
    mask = enet.coef_ != 0
    new_data = data[:,mask]
    return new_data,mask	
    
    
data_=pd.read_csv(r'ALL.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(3846),1))
label2=np.zeros((int(4175),1))
label=np.append(label1,label2)
shu=scale(data)
data_2,index=elasticNet(shu,label)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('ALL_Elastic_net.csv')

