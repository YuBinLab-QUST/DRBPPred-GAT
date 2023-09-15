# DRBPPred-GAT
DRBPPred-GAT: accurate prediction of DNA-binding proteins and RNA-binding proteins based on graph multi-head attention network

##Guiding principles:
**The dataset contains both training dataset and independent test set.

**feature extraction:

CT.py implements CT. 
CTDC.py, CTDD.py and CTDT.py implement CTD. 
EBGW.py implements EBGW.
GTPC.py implements GTPC.
MMI.py implements MMI.
NMBroto.py implements NMBroto.
PseAAC.py implements of PseAAC.
PsePSSM.m implements PsePSSM.

**feature selection:

Autoencoder.py and DenoisingAutoencoder.py implement AE.
EN.py implements EN.
ET.py implements ET.
LASSO.py implements LASSO.
LLE.py implements LLE.
LR.py implements LR.
MI.py implements MI.
SE.py implements SE.
XGBoost.py implements XGBoost.

**Classifier:

NB.py implements NB.
AdaBoost.py implements AdaBoost.
LR.py implements LR.
LightGBM.py implements LightGBM.
KNN.py implements KNN.
SVM.py implements SVM.
CNN.py implements CNN.
LSTM.py implements LSTM.
DNN.py implements DNN.
GRU.py implements GRU.
GAT.py and module.py implement GAT.


