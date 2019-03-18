import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 1, 1)
        self.conv2 = nn.Conv2d(12, 24, 3, 1, 1)
        self.conv3 = nn.Conv2d(24, 48, 5, 1, 2)
        self.conv4 = nn.Conv2d(48, 64, 5, 1, 2)
        self.fc1 = nn.Linear(64*25*25, 1800)
        self.fc2 = nn.Linear(1800, n)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        #print(x.size())  #b*64*100*100  bcwh
        x = x.view(-1, 64*25*25)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x





