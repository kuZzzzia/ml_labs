# download mnist dataset

from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

# checking dataset

import matplotlib.pyplot as plt
plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()

from torch.utils.data import DataLoader

train_data_loader = DataLoader(train_data, 
                               batch_size=1000, 
                               shuffle=True, 
                               num_workers=1)
test_data_loader = DataLoader(test_data, 
                              batch_size=10000, 
                              shuffle=True, 
                              num_workers=1)



import torch.nn as nn
from torch import optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=3               
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 3),     
            nn.ReLU(),                      
            nn.MaxPool2d(2, 2),                
        )
        self.out = nn.Linear(32 * 5 * 5, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

cnn = CNN()

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.01) 


from torch.autograd import Variable
import torch

def test():
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_data_loader:
            test_output = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            return accuracy

def train(num_epochs, cnn, loader):
    cnn.train()
    total_step = len(loader)
    xs = [i + 1 for i in range(num_epochs)]
    ys_acc = []
    ys_loss = []    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, labels) in enumerate(loader):
            
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)          
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                epoch_loss = loss.item()
        epoch_accuracy = test()
        ys_acc.append(epoch_accuracy)
        ys_loss.append(epoch_loss)
        cnn.train()
    # plot('Loss', xs, [(ys_loss, 'loss')])
    plot('Accuracy', xs, [(ys_acc, 'acc')])

import matplotlib.pyplot as plt

def plot(title, xs, ys):
  fig = plt.figure()
  plt.figure().clear()
  for val, lab in ys:
      plt.plot(xs, val, label=lab)
  plt.grid()
  plt.xlabel('epochs')
  plt.title(title)
  plt.tight_layout()
  plt.show()



train(5, cnn, train_data_loader)
print('Accuracy: {}'.format(test()))


