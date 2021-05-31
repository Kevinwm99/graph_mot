import torch
import pandas as pd
from create_dataloader import MOTGraph
from pack import MOTSeqProcessor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
from torch_geometric.utils import to_scipy_sparse_matrix
from collections import OrderedDict
from non_local_embedded_gaussian import NONLocalBlock2D
from generator import GCNStack, Generator
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from utils.graph import compute_edge_feats_dict
device = torch.device("cuda:7")

dataset_para = {'det_file_name': 'frcnn_prepr_det',
                'node_embeddings_dir': 'resnet50_conv',
                'reid_embeddings_dir': 'resnet50_w_fc256',
                'img_batch_size': 5000,  # 6GBytes
                'gt_assign_min_iou': 0.5,
                'precomputed_embeddings': True,
                'overwrite_processed_data': False,
                'frames_per_graph': 'max',  # Maximum number of frames contained in each graph sampled graph
                'max_frame_dist': max,
                'min_detects': 25,  # Minimum number of detections allowed so that a graph is sampled
                'max_detects': None,
                'edge_feats_to_use': ['secs_time_dists', 'norm_feet_x_dists', 'norm_feet_y_dists',
                                      'bb_height_dists', 'bb_width_dists', 'emb_dist'],
                }
cnn_params = {
    'model_weights_path': '/home/kevinwm99/MOT/mot_neural_solver/output/trained_models/reid/resnet50_market_cuhk_duke.tar-232'
}
DATA_ROOT = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train'
DATA_PATH = '/home/kevinwm99/MOT/mot_neural_solver/data'
mot17_seqs = [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 5, 9, 10, 11, 13)]
mot17_train = mot17_seqs[:5]
mot17_val = mot17_seqs[5:]


def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
# https://stackoverflow.com/questions/51387361/pad-a-numpy-array-with-random-values-within-a-given-range
def random_pad(vec, pad_width, *_, **__):
    vec[:pad_width[0]] = np.random.randint(20, 30, size=pad_width[0])
    vec[vec.size-pad_width[1]:] = np.random.randint(30,40, size=pad_width[1])

class Node_Model(nn.Module):
    def __init__(self, dropout=0.0):
        super(Node_Model, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=64,
                                             kernel_size=(1, 3),
                                             bias=True),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                             out_channels=32,
                                             kernel_size=(1, 3),
                                             bias=True),
                                    nn.BatchNorm2d(num_features=32),
                                    nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
                                             out_channels=1,
                                             kernel_size=(1, 2),
                                             bias=True),
                                   nn.BatchNorm2d(num_features=1),
                                   nn.LeakyReLU())
    def forward(self, agg_feat):
        x = self.conv1(agg_feat)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = x.view(1,256)
        return x
class Edge_Model(nn.Module):
    def __init__(self, in_features,
                 num_features=256,
                 ratio=[2, 2, 1, 1],
                 dropout=0.0):
        super(Edge_Model, self).__init__()
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            # set layer
            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

    def forward(self, node_feat):
        node_i = node_feat.unsqueeze(0).unsqueeze(2)
        node_j = torch.transpose(node_i, 1, 2)
        node_ij = torch.abs(node_i - node_j)
        node_ij = torch.transpose(node_ij, 1, 3)
        out_feat = torch.sigmoid(self.sim_network(node_ij))

        return out_feat
if __name__ == '__main__':

    #################################################################################################################################################
    #################################################################################################################################################
    for seq_name in mot17_train:

        processor = MOTSeqProcessor(DATA_ROOT, seq_name, dataset_para, device=device)
        df,frames = processor.load_or_process_detections()
        df_len = len(df)
        max_frame_per_graph = 15
        fps = df.seq_info_dict['fps']
        for i in range(1, len(frames)-max_frame_per_graph+2):


            mot_graph_past = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                 start_frame=i,
                                 end_frame=i+13)
            mot_graph_future = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                 start_frame=i+14,
                                 end_frame=i+14)
            mot_graph_gt = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                 start_frame=i,
                                 end_frame=i+14)

            node_gt,_ = mot_graph_gt._load_appearance_data()
            edge_ixs_gt = mot_graph_gt._get_edge_ixs()
            l1, l2 = (zip(*sorted(zip(edge_ixs_gt[0].numpy(), edge_ixs_gt[1].numpy()))))
            edge_ixs_gt = (torch.tensor((l1, l2)))

            node_past, _ = mot_graph_past._load_appearance_data() # node feature
            edge_ixs_past = mot_graph_past._get_edge_ixs()
            l1, l2 = (zip(*sorted(zip(edge_ixs_past[0].numpy(), edge_ixs_past[1].numpy()))))
            edge_ixs = (torch.tensor((l1, l2)))

            node_fut, _ = mot_graph_future._load_appearance_data()
            mot_graph_current_df = pd.concat([mot_graph_past.graph_df, mot_graph_future.graph_df]).reset_index(drop=True).drop(['index'], axis=1)

            edge_ixs_past = to_scipy_sparse_matrix(edge_ixs_past).toarray()
            edge_current_coo  = F.pad(torch.from_numpy(edge_ixs_past),
                             (0, node_fut.shape[0], 0, node_fut.shape[0]), mode='constant', value=1) # after padding
            edge_current = (from_scipy_sparse_matrix(coo_matrix(edge_current_coo.cpu().numpy()))[0]) # to [2, num edges] torch tensor
            node_current = torch.cat((node_past,node_fut))

            # calculate edge attributes
            edge_feats_dict = compute_edge_feats_dict(edge_ixs=edge_current, det_df=mot_graph_current_df,
                                                      fps=fps,
                                                      use_cuda=True)
            edge_feats = [edge_feats_dict[feat_names] for feat_names in dataset_para['edge_feats_to_use'] if
                          feat_names in edge_feats_dict]
            edge_feats = torch.stack(edge_feats).T
            emb_dists = []
            # divide in case out of memory
            for i in range(0, edge_current.shape[1], 50000):
                emb_dists.append(F.pairwise_distance(node_current[edge_current[0][i:i + 50000]],
                                                     node_current[edge_current[1][i:i + 50000]]).view(-1, 1))

            emb_dists = torch.cat(emb_dists, dim=0)

            # Add embedding distances to edge features if needed
            if 'emb_dist' in dataset_para['edge_feats_to_use']:
                edge_feats = torch.cat((edge_feats.to(device), emb_dists.to(device)), dim=1)


            # print("Edge features", edge_feats.shape)
            print("Edge features", emb_dists.shape)
            # edge_attr = torch.cat((edge_feats, edge_feats))
            edge_attr = torch.cat((emb_dists, emb_dists))
            print(edge_attr.shape)
            print(edge_current.shape)
            print(node_current.shape)
            row, col = edge_current
            print(edge_attr[row].shape, edge_attr[col].shape)
            print(node_current[row].shape, node_current[col].shape)


            neighbors_node = {}
            neighbors_edge ={}
            node_model = Node_Model()
            # edge_model = Edge_Model()
            # edge_model = edge_model.to(device)
            node_model = node_model.to(device)
            for i in range(len(row)): # for all nodes
                if int(row[i]) not in neighbors_node:
                    neighbors_node[int(row[i])] = []
                    neighbors_edge[int(row[i])] = []
                neighbors_edge[int(row[i])].append(edge_attr[int(col[i])])
                neighbors_node[int(row[i])].append(node_current[int(col[i])])
            # print(len(neighbors_edge))
            ############################################################################################
            # node update
            for i in range(len(node_current)):
                neighbors_node_feat = torch.cat(neighbors_node[i], dim=0).view(len(neighbors_node[i]), -1)
                neighbors_edge_feat = torch.cat(neighbors_edge[i], dim=0).view(len(neighbors_edge[i]), -1)
                # aggregate features
                agg_feat = torch.mm(neighbors_node_feat.T.to(device), neighbors_edge_feat.to(device))
                # print(agg_feat.shape)
                node_current[i] = node_model(agg_feat.to(device).unsqueeze(0).unsqueeze(0))
                # break
            # edge update
            new_neighbors_node_feat = {}
            for i in range(len(row)):
                if int(row[i]) not in new_neighbors_node_feat:
                    new_neighbors_node_feat[int(row[i])] = []
                new_neighbors_node_feat[int(row[i])].append(node_current[int(col[i])])
            for i in range(len(node_current)):
                pass
            edge_model = Edge_Model(256)
            edge_model = edge_model.to(device)
            out = edge_model(node_current.to(device))
            print(out.shape)
            # node_i = node_current.unsqueeze(0).unsqueeze(2)
            # node_j = torch.transpose(node_i, 1, 2)
            # node_ij = torch.abs(node_i - node_j)
            # node_ij = torch.transpose(node_ij, 1, 3)
            # print(node_ij.shape)
            # force_edge_feat = torch.cat((torch.eye(node_current.size(1)).unsqueeze(0),
            #                             torch.zeros(node_current.size(1), node_current.size(1)).unsqueeze(0)),
            #                            0).unsqueeze(0).repeat(node_current.size(0), 1, 1, 1).to(device)
            # print(node_current.shape)

            break
        break
        # print(mot_graph_past.graph_df)
        # for i in edge_ixs.T:
        #     print(i)
        # print(edge_ixs[:20].T)
        # print(edge_index_short_term[:124])
        # print(frame,'\n', ids,'\n', detection_ids)
        # edge_index_gt = mot_graph._get_edge_ix_gt()
        # l1, l2 = (zip(*sorted(zip(edge_index_gt[0].numpy(), edge_index_gt[1].numpy()))))
        # edge_index_gt = (torch.tensor((l1, l2)))
        # # edge_index_gt = to_scipy_sparse_matrix(edge_index_gt).toarray()
        # # print(edge_index_gt)
        # # break
        # # node = (node[mot_graph.graph_df.sort_values(by=['id', 'frame']).index])
        # if len(node) < 500:
        #     node = torch.cat((node.to(device), torch.zeros((500 - len(node), 256)).to(device)))
        # else:
        #     node = node[:500]
        # node = node.unsqueeze(0).unsqueeze(0).to(device)
        # sparse_matrix = (to_scipy_sparse_matrix(edge_index_gt))
        # labels = sparse_matrix.toarray()
        # labels = torch.from_numpy(labels).float().to(device)
        # if len(labels) < 500:
        #     pad = nn.ZeroPad2d((0, 500 - len(labels), 0, 500 - len(labels)))
        #     labels = pad(labels)
        # else:
        #     labels = labels[:500]



#################################################################################################################################################
#################################################################################################################################################
