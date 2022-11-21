import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
import sklearn.cluster as skl_cluster

from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# You can import whatever standard packages are required
#
# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  
  # write your code here
  # Refer to sklearn data sets
  return X,y

def get_data_circles(n_points=100):
 
  X,y = make_circles(n_samples=n_points,noise=0.05)
  # write your code here
  # Refer to sklearn data sets
 
  # write your code ...
  return X,y

def get_data_mnist():
  from sklearn.datasets import load_digits
  X,y= load_digits()
  
  
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  Km = skl_cluster.KMeans(n_clusters=k)
  Km.fit(X)
  
 
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  # this is the KMeans object
  # write your code ...
  return Km

def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h = homogeneity_score(ypred_1,ypred_2)
  c = completeness_score(ypred_1,ypred_2)
  v = v_measure_score(ypred_1,ypred_2)
  
  #h,c,v=sklearn.metrics.homogeneity_completeness_v_measure(ypred1, ypred2)
  # refer to sklearn documentation for homogeneity, completeness and vscore
  #h,c,v = 0,0,0 # you need to write your code to find proper values
  return h,c,v
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
def get_data_mnist():
  X,y = load_digits(return_X_y=True)
  #print(X.shape)
  return X,y
def build_lr_model(X=None, y=None):
  
  lr_model = LogisticRegression(random_state=0,max_iter=1000).fit(X, y)
  return lr_model

def build_rf_model(X=None, y=None):
  
  rf_model = RandomForestClassifier(n_estimators=100).fit(X,y)
  
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model

def get_metrics(model=None,X=None,y=None):
  
  y_pred = model.predict(X)
  acc = accuracy_score(y, y_pred)
  print('Accuracy: %f' % accuracy)
  # precision tp / (tp + fp)
  prec = precision_score(y, y_pred)
  print('Precision: %f' % precision)
  # recall: tp / (tp + fn)
  rec = recall_score(y, y_pred)
  print('Recall: %f' % recall)
  # f1: 2 tp / (2 tp + fp + fn)
  f1 = f1_score(y, y_pred)
  print('F1 score: %f' % f1)
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  #acc, prec, rec, f1, auc = 0,0,0,0,0
  # write your code here...
  return acc, prec, rec, f1, auc
