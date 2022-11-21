import torch
from torch import nn
import torch.optim as optim
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import load_digits
import sklearn.cluster as skl_cluster
import sklearn.datasets.samples_generator as skl_smpl
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score

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
  X, y = None
  # write your code ...
  return X,y

def get_data_mnist():
  from sklearn.datasets import load_digits
  X,y= load_digits()
  pass
  
  # write your code ...
  return X,y

def build_kmeans(X=None,k=10):
  Km = skl_cluster.KMeans(n_clusters=k)
  Km.fit(data)
  
 
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
  # this is the KMeans object
  # write your code ...
  return km

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
