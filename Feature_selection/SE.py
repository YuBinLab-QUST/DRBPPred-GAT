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
#import utils.tools as utils
from sklearn.manifold import SpectralEmbedding 
#from  L1_Matine import logistic_dimension
#from dimension_reduction import elasticNet
#from dimension_reduction import KPCA,LLE,SE,TSVD

def SE(data,n_components=300):
    embedding = SpectralEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed

data_=pd.read_csv(r'ALL.csv')
data=np.array(data_)
data=data[:,2:]
[m1,n1]=np.shape(data)
label1=np.ones((int(3846),1))#Value can be changed
label2=np.zeros((int(4175),1))
label=np.append(label1,label2)
shu1=scale(data)
#data_2,mask=logistic_dimension(shu,label,parameter=1)
# date_2,mask=elasticNet(shu1,label1,alpha =0.03,l1_ratio=0.1)
data_2=SE(shu1,n_components=300)

shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('ALL_SE.csv')