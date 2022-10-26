import torch
from torch import nn
import torch.nn.functional as F
def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class CS21M012(nn.Module):
  def __init__(self):
        super().__init__()
        self.m = nn.Softmax(dim =1)
        self.fc1 = nn.Linear(28*28*1, 120)
        self.fc2 = nn.Linear(120, 10)

  def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.m(x)
  # ... your code ...
  # ... write init and forward functions appropriately ...
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
  model = CS21M012()
  input_size=trainX.shape[1]
  model = Sequential([
                    Dense(200, input_shape=(input_size,), activation="relu"),
                    Dense(200, activation='relu'),
                    Dense(10, activation="softmax")])
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
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = CS21M012()
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(CS21M012().parameters(), lr=1e-3)
  return model


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
def test_model(model1=None, test_data=None):
    classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
    model1.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model1(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        #print(f'Predicted: "{predicted}", Actual: "{actual}"')
        
    print('Returning metrics... (rollnumber: xx)')
  
    return predicted,actual

