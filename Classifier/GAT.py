# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 20:14:46 2021

@author: 菜菜
"""

from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
from module import GAT
import torch.nn.functional as F
import pandas as pd
import dgl
import dgl.nn as dglnn
import random
from sklearn.model_selection import KFold
import math
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc
from scipy import interp
import os
import matplotlib.pyplot as plt
from pylab import *
import utils.tools as utils


def get_shuffle(dataset,label):    
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset,label  


def load_data():
    data_=pd.read_csv(r'ALL_Auto.csv')
    data1=np.array(data_)
    data=data1[:,1:]
    [m1,n1]=np.shape(data)
    label1=np.ones((int(3846),1))
    label2=np.zeros((int(4175),1))
    labels=np.append(label1,label2)
    shu=data
    X,y=get_shuffle(shu,labels)
    features = torch.FloatTensor(X)
    labels = torch.squeeze(y)
    g = dgl.knn_graph(features, 5, algorithm='bruteforce-blas', dist='cosine')
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, features, labels

g, features, labels = load_data()


sepscores = []
sepscores_ = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

[sample_num,input_dim]=np.shape(features)
out_dim=2
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
probas_cnn=[]
tprs_cnn = []
sepscore_cnn = []



Kfold = KFold(n_splits = 10, random_state = False)
index = Kfold.split(X=features ,y=labels)
dur = []


for train_index,test_index in index:
    net = GAT(g,in_dim=features.size()[1],num_layers=2,
              num_hidden=8,num_classes=2,
              heads=torch.tensor([32,32,32]),activation=F.relu,
              feat_drop=0,attn_drop=0,
              negative_slope=0,                          #LeakyReLU角度  默认0.2
              residual=False)   
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(180):
        t0 = time.time()

        logits = net(features)
        m = nn.LogSoftmax(dim=1)
        criteria = nn.NLLLoss()
        loss = criteria(m(logits[train_index,:]), labels[train_index])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#        if epoch >= 3:
        dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), np.mean(dur)))
        
    a=logits[test_index]
    probas = F.softmax(a, dim=1)   
    y_class= utils.categorical_probas_to_classes(probas)    
    y_test=utils.to_categorical(labels[test_index])#generate the test 
    ytest=np.vstack((ytest,y_test))
    y_test_tmp=labels[test_index]  
    yscore=np.vstack((yscore,probas))
    
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class,labels[test_index])
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(labels[test_index], probas[:, 1])
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
data_csv.to_csv('GAT.csv')     

    
