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
#from dimensional_reduction import selectFromExtraTrees

def selectFromExtraTrees(data,label):
    clf = ExtraTreesClassifier(n_estimators=1, criterion='gini', max_depth=None, 
                              class_weight=None)#entropy )#entropy
    clf.fit(data,label)
    importance=clf.feature_importances_ 
    model=SelectFromModel(clf,prefit=True)
    new_data = model.transform(data)
    return new_data,importance


#def selectFromExtraTrees(data,label):
#    clf = ExtraTreesClassifier(n_estimators=1, criterion='gini', max_depth=None, 
#                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#                               max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
#                               min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, 
#                               random_state=None, verbose=0, warm_start=False, class_weight=None)#entropy
#    clf.fit(data,label)
#    importance=clf.feature_importances_ 
#    model=SelectFromModel(clf,prefit=True)
#    new_data = model.transform(data)
#    return new_data,importance

data_=pd.read_csv(r'ALL.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(3846),1))#Value can be changed
label2=np.zeros((int(4175),1))
label=np.append(label1,label2)
shu=scale(data)	
data_2,importance=selectFromExtraTrees(shu,label)
shu=data_2
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('ALL_ET.csv')