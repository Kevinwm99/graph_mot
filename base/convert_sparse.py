from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

root = '/home/kevinwm99/MOT/GCN/base/data/ModelNet_10/'

if __name__ =='__main__':

    # data = ModelNet(root,'10')
    pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    train_dataset = ModelNet(root, '10', True, transform, pre_transform)
    test_dataset = ModelNet(root, '10', False, transform, pre_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=6)
    for x in train_loader:
        print(x)
        break
