import torch
import os
import os.path as osp
import init_paths
import numpy as np
from opts import opts
import glob
from torch_geometric.data import Data
import networkx as nx
# from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
device = torch.device("cuda:4")
root = '/home/coin/datasets/MOT2017/MOT17/images/train/MOT17-04-SDP/img1/'
def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        # print(i)
        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v,weight=values['edge_weight'][i])

    return G
def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        labels = nx.get_edge_attributes(G, 'weight')

        # pos = nx.get_node_attributes(G, 'x')
        # pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos=nx.spring_layout(G,seed=2), with_labels=True,
                         node_color=color, cmap="Set2")
        nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G,seed=2), edge_labels=labels)

        # nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
        #                  node_color=color, cmap="Set2")

    plt.show()
    plt.savefig("/home/kevinwm99/MOT/graph879.png")

def cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a)
        b = np.asarray(b) / np.linalg.norm(b)
    return 1. - np.dot(a, b.T)
if __name__ == '__main__':
    embed_list = (sorted(glob.glob(osp.join(root,'*.pt'))))
    print("initialize frame, edge index, node feature, edge weight")
    frame,edge_idx_total,node_feature_total,edge_weight_total = {},{},{},{}
    embed_total =[]
    for emb in (embed_list):
        embed_total.append(torch.load(emb))

    for emb in (embed_total):
        for i, feat in emb.items():
            frame[int(i)] = []
            edge_idx_total[int(i)] = []
            node_feature_total[int(i)] = []
            edge_weight_total[int(i)] = []

    for n,emb in enumerate(embed_list):
        frame_num = int(emb[::-1][emb[::-1].find('.')+1:emb[::-1].find('/')][::-1]) # get frame id
        embed = embed_total[n]
        for i, feat in embed.items():
            frame[int(i)].append(frame_num)
            node_feature_total[int(i)].append(feat.numpy())

    for object, frame_id in frame.items():
        frame_id = torch.tensor(frame_id)
        step = torch.abs(frame_id.reshape(-1,1)-frame_id.reshape(1,-1))
        step = step==1
        r,c = torch.where(step)
        edge_idx_total[object].append((torch.stack((r,c))))
        # edge_weight_total[object]=[]
        for i,j in zip(r,c):
            weight=cosine_distance(node_feature_total[object][i],node_feature_total[object][j])
            # if weight not in edge_weight_total[object]:
            edge_weight_total[object].append(weight)
    # construct one graph for id 909
    ID = 902
    edge_ix = torch.from_numpy(np.asarray(edge_idx_total[ID][0])).long()
    edge_weight = np.round(torch.from_numpy(np.asarray(edge_weight_total[ID])).numpy(),4)
    x = torch.from_numpy(np.asarray(node_feature_total[ID]))
    y = torch.from_numpy(np.asarray([ID]*len(x)))
    # edge_weight_ = [x for pair in zip(edge_weight,edge_weight) for x in pair]
    print(sorted(frame))
    print(len(edge_weight))
    print(len(x))
    data = Data(x=x,edge_index = edge_ix, edge_weight=edge_weight,y=y)
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Number of edge features: {data.num_edge_features}')

    G = to_networkx(data, to_undirected=True,edge_attrs=['edge_weight'])
    visualize(G, color=data.y)