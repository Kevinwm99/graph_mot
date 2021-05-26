from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

if __name__ == '__main__':
    dataset = 'Cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    root = '/home/kevinwm99/MOT/GCN/base/data'
    dataset = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data)

    print(data.num_features)
    print(data.x.shape)
    print(data.train_pos_edge_index.shape)