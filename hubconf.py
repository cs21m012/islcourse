from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import torch
from torch import nn

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

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

	
      

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
  #model.compile(loss='crossentropy', optimizer='SGD', metrics=['accuracy'])
  


  error = nn.CrossEntropyLoss()

  learning_rate = lr
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
  yhat_probs = model1.predict(testX, verbose=0)
  yhat_classes = mode1l.predict_classes(testX, verbose=0)
  yhat_probs = yhat_probs[:, 0]
  yhat_classes = yhat_classes[:, 0]
  accuracy_val = accuracy_score(testy, yhat_classes)
  precision_val = precision_score(testy, yhat_classes)


  recall = recall_score(testy, yhat_classes)
  f1score_val = f1_score(testy, yhat_classes)

  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: CS21M012)')
  
  return accuracy_val, precision_val, recall_val, f1score_val


