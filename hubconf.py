
import torch
from torch import nn

import torch
import torchmetrics
import torchmetrics.functional

import torch.optim as optim
from torch import nn
#from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchmetrics.classification import MulticlassF1Score

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class CS21M012(nn.Module):
    def __init__(self):
        super(CS21M012, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    metrics = torchmetrics.MetricCollection([
            # Accuracy: due to mode multiclass, not multilabel, this uses same formula as Precision
            torchmetrics.Accuracy(num_classes=10),
            torchmetrics.Precision(num_classes=10),
            torchmetrics.Recall(num_classes=10),
            #torchmetrics.F(num_classes=10),
        ])
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

	
metrics = torchmetrics.MetricCollection([
            # Accuracy: due to mode multiclass, not multilabel, this uses same formula as Precision
            torchmetrics.Accuracy(num_classes=10),
            torchmetrics.Precision(num_classes=10),
            torchmetrics.Recall(num_classes=10),
            #torchmetrics.F(num_classes=10),
        ])     

  # ... your code ...
  # ... write init and forward functions appropriately ...
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  
  model = CS21M012()
  model.add(Conv2D(32, (5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
 
  model.add(MaxPooling2D((2,2)))
  model.add(Conv2D(64, (5, 5), activation='relu', kernel_initializer='he_uniform'))
  
  model.add(MaxPooling2D((2,2)))
  model.add(Flatten())
 
  model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # Use softmax and cross entropy loss functions
  # set model variable to proper object, make use of train_data
  
  print ('Returning model... (rollnumber: CS21M012)')
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = CS21M012()
 

  error = nn.CrossEntropyLoss()

  learning_rate = 0.001
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
 
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
  
  
  
  print ('Returning model... (rollnumber: CS21M012)')
  
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):
  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
   
  num_epochs = 5
  count = 0
  # Lists for visualization of loss and accuracy 
  loss_list = []
  iteration_list = []
  accuracy_list = []

   # Lists for knowing classwise accuracy
  predictions_list = []
  labels_list = []

  for epoch in range(num_epochs):
      for images, labels in train_loader:
         # Transfering images and labels to GPU if available
          images, labels = images.to(device), labels.to(device)
    
          train = Variable(images.view(100, 1, 28, 28))
          labels = Variable(labels)
        
        # Forward pass 
          outputs = model(train)
          loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
          optimizer.zero_grad()
        
        #Propagating the error backward
          loss.backward()
        
        # Optimizing the parameters
          optimizer.step()
    
          count += 1
          if not (count % 50):    # It's same as "if count % 50 == 0"
            total = 0
            correct = 0
        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
            
                test = Variable(images.view(100, 1, 28, 28))
            
                outputs = model(test)
            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
            
                total += len(labels)
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy) 
        
   #if not (count % 500):
            #print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
  
  #accuracy_val = torchmetrics.Accuracy(10)
  #precision_val = torchmetrics.Precision(10)


  #recall = torchmetrics.Recall(10)
  #score_val = torchmetrics.F1Score(10)

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: CS21M012)')
  
  return accuracy_list


