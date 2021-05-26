# from non_local_embedded_gaussian import NONLocalBlock2D
# from non_local_concatenation import NONLocalBlock2D
from non_local_dot_product import NONLocalBlock2D
# from non_local_gaussian import NONLocalBlock2D
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseGCNConv

import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
device = torch.device("cuda:7")
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_1 = NONLocalBlock2D(in_channels=64)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
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
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=500, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256,out_features=500),
            nn.ReLU(),)
        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 1, out_features = 500),
            nn.ReLU()
        )
        self.gconv1 = GCNConv(256, 512)
        self.gconv2 = GCNConv(512, 1024)
        # self.gconv3 = GCNConv(32, 16)
        # self.gconv4 = GCNConv(16, 8)
        self.glinear1 = DenseGCNConv(1024,500)

    def forward(self, x):
        batch_size = x.size(0)
        data = x.squeeze(0).squeeze(0)
        # x= x.unsqueeze(0)
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

        output = self.fc1(output.view( batch_size,-1))
        edge_ix= self.fc2(output.view(-1,batch_size))
        # print(edge_ix)
        # mask = torch.zeros_like(edge_ix)
        # mask[edge_ix >= 0.5] = 1  # apply threshold
        sparse = (coo_matrix(edge_ix.detach().cpu()))
        sparse = from_scipy_sparse_matrix(sparse)
        sparse = torch.clone(sparse[0]).requires_grad_(False).to(device)
        x = F.relu(self.gconv1(data, sparse))
        # print(x.shape)
        x = F.relu(self.gconv2(x, sparse))
        # x = F.relu(self.gconv3(x, sparse))
        # x = F.relu(self.gconv4(x, sparse))
        x = F.relu(self.glinear1(x, edge_ix,add_loop=False))
        return x

class GCNStack(nn.Module):
    def __init__(self):
        super(GCNStack,self).__init__()

        self.gconv1 = GCNConv(256,256)
        self.gconv2 = GCNConv(256,500)
        # self.conv3 = GCNConv(64,500)
    def forward(self,x):
        feat, edge_ix = x.data, x.edge_ix
        print(feat.shape)
        print(edge_ix.shape)
        x = F.relu(self.conv1(feat,edge_ix))
        print(x.shape)
        x = F.relu(self.conv2(x,edge_ix))
        print(x.shape)
        # x = F.relu(self.conv3(x,edge_ix))
        # print(x.shape)
        return x
