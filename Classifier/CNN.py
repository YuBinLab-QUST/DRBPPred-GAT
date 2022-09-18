#import tensorflow as tf 

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
import re
from scipy import interp
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Input,Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
from sklearn.preprocessing import scale,StandardScaler
from keras.layers import Dense, merge,Input,Dropout
from keras.models import Model

def to_class(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y
# Origanize data
def get_shuffle(dataset,label):    
    #shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label 
# Origanize data

data_=pd.read_csv(r'ALL_Auto.csv')
data=np.array(data_)
data=data[:,1:]
[m1,n1]=np.shape(data)
label1=np.ones((int(3846),1))#Value can be changed
label2=np.zeros((int(4175),1))
label=np.append(label1,label2)
X_=scale(data)
y_= label
X,y=get_shuffle(X_,y_)
sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5


def get_CNN_model(input_dim,out_dim):
    model = Sequential()
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    model.add(Conv1D(filters = 32, kernel_size = 3, padding = 'same', activation= 'relu'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))
    #model.add(MaxPooling1D(pool_size=2,strides=1,padding="SAME"))  
    model.add(Flatten())
    model.add(Dense(int(input_dim/2), activation = 'relu'))
    model.add(Dense(int(input_dim/4), activation = 'relu'))
    model.add(Dense(out_dim, activation = 'softmax',name="Dense_2"))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics =['accuracy'])
    return model

[sample_num,input_dim]=np.shape(X)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []
skf= StratifiedKFold(n_splits=10)
for train, test in skf.split(X,y):
    clf_cnn = get_CNN_model(input_dim,out_dim)
    X_train_cnn=np.reshape(X[train],(-1,1,input_dim))
    X_test_cnn=np.reshape(X[test],(-1,1,input_dim))
    clf_list = clf_cnn.fit(X_train_cnn, to_categorical(y[train]),nb_epoch=20)   
    y_cnn_probas=clf_cnn.predict(X_test_cnn)
    probas_cnn.append(y_cnn_probas)
    y_class= utils.categorical_probas_to_classes(y_cnn_probas)
    
    y_test=utils.to_categorical(y[test])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=y[test]  
    yscore=np.vstack((yscore,y_cnn_probas))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class,y[test])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(y[test], y_cnn_probas[:, 1])
    tprs_cnn.append(interp(mean_fpr, fpr, tpr))
    tprs_cnn[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    sepscore_cnn.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('NB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))

row=ytest.shape[0]
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_sum.csv')

yscore_=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore_)
yscore_sum.to_csv('yscore_sum.csv')

scores=np.array(sepscore_cnn)
result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscore_cnn.append(H1)
result=sepscore_cnn
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('CNN.csv')

