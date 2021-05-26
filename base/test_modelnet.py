from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import torch
from scipy.sparse import coo_matrix
root = '/home/kevinwm99/MOT/GCN/base/data/ModelNet_10/'

if __name__ =='__main__':
    edge_index = torch.from_numpy(np.array([[0,1,2,3],
                  [1,2,3,4]]))
    sparse_matrix = (to_scipy_sparse_matrix(edge_index))
    # coo_matrix(sparse_matrix)
    print(sparse_matrix.toarray())