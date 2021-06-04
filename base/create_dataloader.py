
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import configparser
from lapsolver import solve_dense

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from models.resnet import resnet50_fc256, load_pretrained_weights
from utils.rgb import BoundingBoxDataset,load_precomputed_embeddings,load_embeddings_from_imgs
from utils.iou import iou
from utils.graph import get_time_valid_conn_ixs, get_knn_mask, compute_edge_feats_dict

from pack import get_mot_det_df, get_mot_det_df_from_gt,MOTSeqProcessor
# device = torch.device("cuda:6")

#######################################  config ############################$$$$$$$$$$$$$$##############################
dataset_para={   'det_file_name': 'frcnn_prepr_det',
                'node_embeddings_dir': 'resnet50_conv',
                'reid_embeddings_dir': 'resnet50_w_fc256',
                 'img_batch_size': 5000 , # 6GBytes
                'gt_assign_min_iou': 0.5,
                'precomputed_embeddings': True,
                'overwrite_processed_data': False,
                'frames_per_graph': 'max', # Maximum number of frames contained in each graph sampled graph
                  'max_frame_dist': max,
                'min_detects': 25, # Minimum number of detections allowed so that a graph is sampled
                  'max_detects': 500,
             }
cnn_params={
    'model_weights_path': '/home/kevinwm99/MOT/mot_neural_solver/output/trained_models/reid/resnet50_market_cuhk_duke.tar-232'
    }
DATA_ROOT = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train'
DATA_PATH = '/home/kevinwm99/MOT/mot_neural_solver/data'
_SEQ_TYPES = {}
# Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside img
# hence we crop its detections to also be inside it)
_ENSURE_BOX_IN_FRAME = {'MOT': False,
                        'MOT_gt': False,
                        'MOT15': True,
                        'MOT15_gt': False}

# MOT17 Sequences
mot17_seqs = [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in ('DPM', 'SDP', 'FRCNN', 'GT')]
# mot17_seqs += [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in ('DPM', 'SDP', 'FRCNN')]
for seq_name in mot17_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT'

# We now map each sequence name to a sequence type in _SEQ_TYPES
_SEQ_TYPE_DETS_DF_LOADER = {'MOT': get_mot_det_df,
                            'MOT_gt': get_mot_det_df_from_gt,}
DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')
########################################################################################################################

class Graph(Data):
    """
    This is the class we use to instantiate our graph objects. We inherit from torch_geometric's Data class and add a
    few convenient methods to it, mostly related to changing data types in a single call.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _change_attrs_types(self, attr_change_fn):
        """
        Base method for all methods related to changing attribute types. Iterates over the attributes names in
        _data_attr_names, and changes its type via attr_change_fun

        Args:
            attr_change_fn: callable function to change a variable's type
        """
        # These are our standard 'data-related' attribute names.
        _data_attr_names = ['x', # Node feature vecs
                           'edge_attr', # Edge Feature vecs
                           'edge_index', # Sparse Adjacency matrix
                           'node_names', # Node names (integer values)
                           'edge_labels', # Edge labels according to Network Flow MOT formulation
                           'edge_preds', # Predicted approximation to edge labels
                           'reid_emb_dists'] # Reid distance for each edge

        for attr_name in _data_attr_names:
            if hasattr(self, attr_name):
                if getattr(self, attr_name ) is not None:
                    old_attr_val = getattr(self, attr_name)
                    setattr(self, attr_name, attr_change_fn(old_attr_val))

    def tensor(self):
        self._change_attrs_types(attr_change_fn= torch.tensor)
        return self

    def float(self):
        self._change_attrs_types(attr_change_fn= lambda x: x.float())
        return self

    def numpy(self):
        self._change_attrs_types(attr_change_fn= lambda x: x if isinstance(x, np.ndarray) else x.detach().cpu().numpy())
        return self

    def cpu(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn= lambda x: x.cpu())
        return self

    def cuda(self):
        #self.tensor()
        self._change_attrs_types(attr_change_fn=lambda x: x.cuda())
        return self

    def to(self, device):
        self._change_attrs_types(attr_change_fn=lambda x: x.to(device))

    def device(self):
        if isinstance(self.edge_index, torch.Tensor):
            return self.edge_index.device

        return torch.device('cpu')

class MOTGraph(object):
    """
    This the main class we use to create MOT graphs from detection (and possibly ground truth) files. Its main attribute
    is 'graph_obj', which is an instance of the class 'Graph' and serves as input to the tracking model.

    Moreover, each 'MOTGraph' has several additional attributes that provide further information about the detections in
    the subset of frames from which the graph is constructed.

    """
    def __init__(self, seq_det_df = None, start_frame = None, end_frame = None, ensure_end_is_in = False, step_size = None,
                 seq_info_dict = None, dataset_params = None, inference_mode = False, cnn_model = None, max_frame_dist = None):
        self.dataset_params = dataset_params
        self.step_size = step_size
        self.seq_info_dict = seq_info_dict
        self.inference_mode = inference_mode
        self.max_frame_dist = max_frame_dist

        self.cnn_model = cnn_model

        if seq_det_df is not None:
            self.graph_df, self.frames, self.ids, self.all_frames = self._construct_graph_df(seq_det_df= seq_det_df.copy(),
                                                                  start_frame = start_frame,
                                                                  end_frame = end_frame,
                                                                  ensure_end_is_in=ensure_end_is_in)

    def _construct_graph_df(self, seq_det_df, start_frame, end_frame = None, ensure_end_is_in = False):
        """
        Determines which frames will be in the graph, and creates a DataFrame with its detection's information.

        Args:
            seq_det_df: DataFrame with scene detections information
            start_frame: frame at which the graph starts
            end_frame: (optional) frame at which the graph ends
            ensure_end_is_in: (only if end_frame is given). Bool indicating whether end_frame must be in the graph.

        Returns:
            graph_df: DataFrame with rows of scene_df between the selected frames
            valid_frames: list of selected frames

        """
        if end_frame is not None:
            # Just load all frames between start_frame and end_frame at the desired step size
            valid_frames = np.arange(start_frame, end_frame + 1, self.step_size)

            if ensure_end_is_in and (end_frame not in valid_frames):
                valid_frames = valid_frames.tolist() + [end_frame]
            if self.dataset_params['max_detects'] is not None:
                # print(self.dataset_params['max_detects'])
                scene_df_ = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
                frames_cumsum = scene_df_.groupby('frame')['bb_left'].count().cumsum()
                valid_frames = frames_cumsum[frames_cumsum <= self.dataset_params['max_detects']].index


        else:
            # Consider all posible future frames (at distance step_size)
            valid_frames = np.arange(start_frame, seq_det_df.frame.max(), self.step_size)
            # print("valid")
            # print(valid_frames)
            # We cannot have more than dataset_params['frames_per_graph'] frames
            if self.dataset_params['frames_per_graph'] != 'max':
                valid_frames = valid_frames[:self.dataset_params['frames_per_graph']]

            # We cannot have more than dataset_params['max_detects'] detections
            if self.dataset_params['max_detects'] is not None:
                print(self.dataset_params['max_detects'])
                scene_df_ = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
                frames_cumsum = scene_df_.groupby('frame')['bb_left'].count().cumsum()
                valid_frames = frames_cumsum[frames_cumsum <= self.dataset_params['max_detects']].index

        graph_df = seq_det_df[seq_det_df.frame.isin(valid_frames)].copy()
        # print(graph_df)
        # graph_df = graph_df.sort_values(by=['id','frame']).reset_index(drop=True)
        # print(graph_df)
        graph_df = graph_df.sort_values(by=['frame', 'detection_id']).reset_index(drop=True)
        # print(graph_df.sort_values(by=['id', 'frame']).reset_index(drop=True))
        return graph_df, sorted(graph_df.frame.unique()),sorted(graph_df.id.values), sorted(graph_df.frame.values)

    def _get_edge_idx(self):
        detection_id = torch.from_numpy(np.array(self.ids))
        # print((self.graph_df.id))
        unique_ids = (self.graph_df.id.unique())
        # print(unique_ids)
        # max_frame_dist = self.dataset_params['frames_per_graph']
        max_frame_dist = 1
        edge_ixs = []
        len_prev_object = 0
        for id_ in unique_ids:
            frame_idx = torch.where(detection_id == id_)[0] + 1
            changepoints = torch.where(frame_idx[1:] != frame_idx[:-1])[0] + 1
            changepoints = torch.cat((changepoints, torch.as_tensor([frame_idx.shape[0]]).to(changepoints.device)))
            all_det_ixs = torch.arange(frame_idx.shape[0], device=frame_idx.device)
            for start_frame_ix, end_frame_ix in zip(changepoints[:-1], changepoints[1:]):
                curr_frame_ixs = all_det_ixs[start_frame_ix: end_frame_ix]
                curr_frame_num = frame_idx[curr_frame_ixs[0]]
                # curr_frame_id = detection_id[curr_frame_ixs[0]]
                past_frames_ixs = torch.where(torch.abs(frame_idx[:start_frame_ix] - curr_frame_num) <= max_frame_dist)[0]

                edge_ixs.append(
                    torch.cartesian_prod(past_frames_ixs + len_prev_object, curr_frame_ixs + len_prev_object))
            len_prev_object += len(frame_idx)
        #
        edge_ixs = torch.cat(edge_ixs).T
        return edge_ixs

    def _get_edge_ixs(self):
        frame_num = self.all_frames
        all_ids = torch.from_numpy(np.array(self.ids))
        detection_ids = self.graph_df['detection_id'].values
        unquie_ids = self.graph_df.id.unique()
        max_frame_distance = 5
        edge_ixs = []

        for id in unquie_ids:
            unquie_connect = torch.from_numpy(np.array(self.graph_df['detection_id'][self.graph_df['id'] == id]))
            frame_idx = np.arange(len(unquie_connect))+1
            frame_idx = torch.from_numpy(frame_idx)
            all_det_ixs = torch.arange(frame_idx.shape[0], device=frame_idx.device)
            # print(all_det_ixs)
            # print(unquie_connect)
            for start_frame_ix, end_frame_ix in zip(frame_idx[:-1], frame_idx[1:]):
                curr_frame_ixs = all_det_ixs[start_frame_ix: end_frame_ix]
                curr_frame_num = frame_idx[curr_frame_ixs[0]]
                # print(curr_frame_ixs)
                # curr_frame_id = detection_id[curr_frame_ixs[0]]
                past_frames_ixs = torch.where(torch.abs(frame_idx[:start_frame_ix] - curr_frame_num) <= max_frame_distance)[0]
                # print(unquie_connect[past_frames_ixs])
                # print(unquie_connect[past_frames_ixs])
                # print(unquie_connect[curr_frame_ixs])
                edge_ixs.append(
                    torch.cartesian_prod(unquie_connect[past_frames_ixs], unquie_connect[curr_frame_ixs]))
                # print(edge_ixs)
                # print(edge_ixs)
        edge_ixs = torch.cat(edge_ixs).T
            # print(edge_ixs)
            # print(changepoints)
            # print((unquie_connect))
        # return frame_num, all_ids, detection_ids
        return edge_ixs

    def _get_edge_ix_gt(self):
        unique_ids = self.graph_df.id.unique()
        detection_id = torch.from_numpy(np.array(self.ids))
        max_frame_dist = 1
        edge_ixs = []
        for id_ in unique_ids:
            frame_idx = (np.array(self.graph_df['detection_id'][self.graph_df['id']==id_]))
            for start_frame_ix, end_frame_ix in zip(frame_idx[:-1], frame_idx[1:]):
                edge_ixs.append([start_frame_ix,end_frame_ix])

        return torch.from_numpy(np.array(edge_ixs).T)

    def _load_appearance_data(self):
        """
        Loads embeddings for node features and reid.
        Returns:
            tuple with (reid embeddings, node_feats), both are torch.tensors with shape (num_nodes, embed_dim)
        """
        if self.inference_mode and not self.dataset_params['precomputed_embeddings']:
            assert self.cnn_model is not None
            print("USING CNN FOR APPEARANCE")
            _, node_feats, reid_embeds = load_embeddings_from_imgs(det_df = self.graph_df,
                                                                    dataset_params = self.dataset_params,
                                                                    seq_info_dict = self.seq_info_dict,
                                                                    cnn_model = self.cnn_model,
                                                                    return_imgs = False,
                                                                    use_cuda = self.inference_mode)

        else:
            # print("load computed reid embeddings")
            reid_embeds = load_precomputed_embeddings(det_df=self.graph_df,
                                                      seq_info_dict=self.seq_info_dict,
                                                      embeddings_dir=self.dataset_params['reid_embeddings_dir'],
                                                      use_cuda=self.inference_mode)
            if self.dataset_params['reid_embeddings_dir'] == self.dataset_params['node_embeddings_dir']:
                # print("load computed node features")
                node_feats = reid_embeds.clone()

            else:
                # print("load computed node features")
                node_feats = load_precomputed_embeddings(det_df=self.graph_df,
                                                          seq_info_dict=self.seq_info_dict,
                                                          embeddings_dir=self.dataset_params['node_embeddings_dir'],
                                                          use_cuda=self.inference_mode)

        return reid_embeds, node_feats

import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,data):
        super(Net, self).__init__()
        self.data = data
        self.num_features = data.shape[1]
        self.num_nodes = data.shape[0]
        self.hidden = 512
        self.linear1 = torch.nn.Linear(self.num_features,self.hidden)
        self.linear2 = torch.nn.Linear(self.hidden,self.hidden)
        self.linear3 = torch.nn.Linear(self.hidden, self.hidden)
        self.linear4 = torch.nn.Linear(self.hidden,self.num_nodes)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)

        return x

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(dataset.num_features, 128)
#         self.conv2 = GCNConv(128, 64)
#
#     def encode(self):
#         x = self.conv1(data.x, data.train_pos_edge_index)
#         x = x.relu()
#         return self.conv2(x, data.train_pos_edge_index)

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_scipy_sparse_matrix

if __name__ == '__main__':
    pass
    ################################## LOAD MODEL #################################################
    # cnn_model = resnet50_fc256(10, loss='xent', pretrained=True)
    # torch.load(cnn_params['model_weights_path'], map_location=device)
    # load_pretrained_weights(cnn_model, cnn_params['model_weights_path'], device=device)
    # cnn_model.return_embeddings = True
    # cnn_model.to(device)
    # ###############################################################################################
    # for seq_name in sorted(mot17_seqs):
    #
    #     processor = MOTSeqPeocessor(DATA_ROOT,seq_name,dataset_para,cnn_model,device=device)
    #     df = processor.load_or_process_detections()
    #     mot_graph = MOTGraph(seq_det_df=df,seq_info_dict=df.seq_info_dict,dataset_params=dataset_para,cnn_model=cnn_model,start_frame=1)
    #     node,_ = mot_graph._load_appearance_data()
    #     edge_index = mot_graph._get_edge_idx()
    #     node = (node[mot_graph.graph_df.sort_values(by=['id','frame']).index])
    #     # print(len(node))
    #
    #
    #     sparse_matrix = (to_scipy_sparse_matrix(edge_index))
    #     labels = sparse_matrix.toarray()
    #     labels = torch.from_numpy(labels).double().float()
    #     criterion = torch.nn.MSELoss()
    #     # criterion = torch.nn.SmoothL1Loss()
    #
    #     model = Net(node)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #     epochs = 50
    #     accum_loss = 0.0
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()
    #         output = model(node)
    #         mask = torch.zeros_like(output)
    #         mask[output>0.5]=1  # apply threshold
    #         mask = torch.tensor(mask,requires_grad=True)
    #         loss = criterion(mask,labels)
    #         loss.backward()
    #         optimizer.step()
    #         print("loss: {}".format(loss))
    #         # print(loss)
    #         accum_loss += loss
    #         print(accum_loss)
    #
    #     break