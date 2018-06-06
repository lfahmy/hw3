import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):
    def __init__(self, isDropOut = True):
        super(classifier, self).__init__()
        ## WRITE YOUR CODE HERE, Define the network structure ##
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        ## WRITE YOUR CODE HERE ##
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2, stride=2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2, stride=2)
        x = F.relu(self.fc1(x), inplace=True)
        if(isDropOut):
        	x = F.dropout2d(x, p=.4)
        x = F.softmax(self.fc2(x))
        return x

