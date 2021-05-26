from non_local_embedded_gaussian import NONLocalBlock2D
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
import torch.nn.functional as F

class Generator(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_1 = NONLocalBlock2D(in_channels=32)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_2 = NONLocalBlock2D(in_channels=64)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.nl_3 = NONLocalBlock2D(in_channels=128)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=125, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(125),
            nn.ReLU(),
            nn.MaxPool2d((2,4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=1, out_features=500),
            nn.ReLU())

    def forward(self, x):
        batch_size = x.size(0)
        feature_1 = self.conv_1(x)
        # print(feature_1.shape)
        nl_feature_1 = self.nl_1(feature_1)
        # print(nl_feature_1.shape)
        feature_2 = self.conv_2(nl_feature_1)
        # print(feature_2.shape)
        nl_feature_2 = self.nl_2(feature_2)
        # print(nl_feature_2.shape)
        # print(self.conv_3(nl_feature_2).view(batch_size, -1))
        output = self.conv_3(nl_feature_2)
        # print(output.shape)
        output = self.conv_4(output)
        # print(output.shape)
        output = self.fc(output.view(-1, batch_size))
        return output

class GCNStack(nn.Module):
    def __init__(self):
        super(GCN_stack,self).__init__()

        self.conv1 = GCNConv(1,32)
        self.conv2 = GCNConv(32,64)
    def forward(self,x):
        feat, edge_ix = x.data, x.edge_ix
        print(feat.shape)
        print(edge_ix.shape)
        x = F.relu(self.conv1(feat,edge_ix))
        print(x.shape)
        x = F.relu(self.conv2(x,edge_ix))
        print(x.shape)
        return x