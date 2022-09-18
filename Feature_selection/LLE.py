import scipy.io as sio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale,StandardScaler
from sklearn.metrics import roc_curve, auc
from dimensional_reduction import LLE
#import utils.tools as utils
data_=pd.read_csv(r'ALL.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(3846),1))#Value can be changed
label2=np.zeros((int(4175),1))
label=np.append(label1,label2)
shu=scale(data)	
new_X=LLE(shu,n_components=300)
shu=new_X
data_csv = pd.DataFrame(data=shu)
data_csv.to_csv('ALL_LLE.csv')
