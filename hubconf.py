import torch
from torch import nn

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score

class CS21M012_Ad(nn.Module):
  def __init__(self,config):
    super(CS21M012_Ad, self).__init__()
    self.layers=nn.ModuleList()
    for x in config:
      self.layers.append(Conv2d(in_channels=x[0],out_channels=x[1],kernel_size=x[2],stride=x[3],padding=x[4]))
    #self.relu=ReLU()
  
  def forward(self, x):
    for conv in self.layers:
      x = conv(x)
      x=F.relu(x)
    x = torch.flatten(x, 1)
    features=x.shape
    fc1=nn.Linear(features[1],10)
    x=fc1(x)
    sm=nn.Softmax(dim = 1)
    x=sm(x)
    return x
  

def train_network(train_loader, optimizer,criteria, e,model):
  for epoch in range(e): 

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        
        inputs, labels = data

      
        optimizer.zero_grad()

        outputs = model(inputs)
      
        tmp = torch.nn.functional.one_hot(labels, num_classes= 10)
        loss = criteria(outputs, tmp)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

  print('Finished Training') 
  
  # ... your code ...
  # ... write init and forward functions appropriately ...
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def loss_fun(y_pred, y_ground):
  v = -(y_ground * torch.log(y_pred + 0.0001))
  v = torch.sum(v)
  return v


def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = CS21M012_Ad(config)    
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  criteria=loss_fun
  train_network(train_data_loader,optimizer,criteria,n_epochs,model)
  return model
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  print ('Returning model... (rollnumber: xx)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)



  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  # In addition,
  # Refer to config dict, where learning rate is given, 
  # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
  # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
  # You need to create 2d convoution layers as per specification above in each element
  # You need to add a proper fully connected layer as the last layer
  
  # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
  # HINT: Flatten function can also be used if required

  
  print ('Returning model... (rollnumber: xx)')
 

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torchmetrics.classification import accuracy
from torchmetrics import Precision, Recall, F1Score, Accuracy

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
  
    accuracy1 = Accuracy()
    print('Accuracy :', accuracy1(pred,y)*100)
    precision = Precision(average = 'macro', num_classes = 10)
    print('precision :', precision(pred,y)*100)

    recall = Recall(average = 'macro', num_classes = 10)
    print('recall :', recall(pred,y)*100 )
    f1_score = F1Score(average = 'macro', num_classes = 10)
    print('f1_score :', f1_score(pred,y)*100)
    return accuracy1,precision, recall, f1_score
