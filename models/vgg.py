import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['VGGNet']


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
        #self.conv3 = nn.Conv2d(64, 256, 6)
        #self.BN3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 4, n_class)  # 6*6 from image dimension
        self.BN4 = nn.BatchNorm1d(n_class)
        #self.fc2 = nn.Linear(64, 15)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.35)
        self.dropout4 = nn.Dropout(0.2)

    def forward(self, x):
        #print(x.shape)
        #x = x.view(-1, 3, 28, 28)
        x = F.max_pool2d(F.relu(self.BN1(self.conv1(x))), 2)
        x = self.dropout1(x)
        #print(x.shape)
        #256, 32, 14, 14
        x = F.max_pool2d(F.relu(self.BN2(self.conv2(x))), 2)
        x = self.dropout2(x)
        #print(x.shape)
        #256, 64, 6, 6
        #x = F.relu(self.BN3(self.conv3(x)))
        #x = self.dropout3(x)
        #print(x.shape)
        #256 64 4 4
        x = x.view(-1, self.num_flat_features(x))
        #print(x.shape)
        #256 1024
        x = F.relu(self.BN4(self.fc1(x)))
        #x = self.dropout4(x)
        #x = F.relu(self.fc2(x))
#         x = self.fc2(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 