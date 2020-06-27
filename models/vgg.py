import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['VGGNet', 'DNN', 'AlexNet']


class AlexNet(nn.Module):
    def __init__(self, n_class):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1152, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, n_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x1))
        x = self.fc3(x)
#         return x, x1
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 


class VGGNet(nn.Module):
    """
    Conv->Relu->Pool -> Conv->Relu->Pool -> Conv->Relu -> Dense->Relu -> Dense->Softmax
    """
    def __init__(self, n_class):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.BN1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 256, 11)
        self.BN2 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4, n_class) 
        self.BN4 = nn.BatchNorm1d(n_class)
        #self.fc2 = nn.Linear(64, 15)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.BN1(self.conv1(x))), 2)
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.BN2(self.conv2(x))), 2)
        x = self.dropout2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.BN4(self.fc1(x))
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 


class DNN(nn.Module):
    def __init__(self, n_class):
        super(DNN, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        size = x.size()
        x = x.view(size[0], -1)
        x = self.dnn(x)
        return x
