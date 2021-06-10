import torch
import matplotlib.pyplot as plt
import pandas as pd
from mot_graph import MOTGraph
from pack import MOTSeqProcessor
from loguru import logger
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.nn import GATConv

dataset_para = {'det_file_name': 'frcnn_prepr_det',
                'node_embeddings_dir': 'resnet50_conv',
                'reid_embeddings_dir': 'resnet50_w_fc256',
                'img_batch_size': 5000,  # 6GBytes
                'gt_assign_min_iou': 0.5,
                'precomputed_embeddings': True,
                'overwrite_processed_data': False,
                'frames_per_graph': 'max',  # Maximum number of frames contained in each graph sampled graph
                'max_frame_dist': 'max',
                'min_detects': 25,  # Minimum number of detections allowed so that a graph is sampled
                'max_detects': 450,
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


class GraphData(torch.utils.data.Dataset):
    def __init__(self, root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=None):
        super(GraphData, self).__init__()
        self.num_node_per_graph = 150
        self.max_frame_per_graph = 5
        self.all_seq_name = all_seq_name
        self.dataset_para = datasetpara
        self.device = device
        self.root = root
        self.share_weight = torch.rand((32, 256))
        self._get_seq_frames_index()

    def _get_seq_frames_index(self):
        self.seq_index = []
        for seq_name in self.all_seq_name:
            processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
            df, frames = processor.load_or_process_detections()
            for i in range(1, len(frames)-self.max_frame_per_graph+2):
                self.seq_index.append([seq_name, i])

    def __len__(self):
        return len(self.seq_index)

    def _dot(self, x, y):

        return torch.dot(torch.dot(self.share_weight, x.T), torch.dot(self.share_weight, y))

    def __getitem__(self, index):
        seq_name, start_frame = self.seq_index[index]
        processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
        df, frames = processor.load_or_process_detections()
        # df_len = len(df)
        # max_frame_per_graph = 15
        fps = df.seq_info_dict['fps']
        print("Construct graph {} from frame {} to frame {}".format(seq_name, start_frame, start_frame + self.max_frame_per_graph))

        mot_graph = MOTGraph(seq_det_df=df,
                             seq_info_dict=df.seq_info_dict,
                             dataset_params=dataset_para,
                             start_frame=start_frame,
                             max_frame_dist=5,
                             end_frame=start_frame + (self.max_frame_per_graph-2))

        node_feat, edge_ixs = mot_graph.load_node_and_edge()
        return node_feat, edge_ixs


class TemporalRelationGraph(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalRelationGraph, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=8, dropout=0.0)

    def forward(self, x, edge_index):
        x, (edge_index, alpha) = (self.conv1(x, edge_index, return_attention_weights=True))
        return x, edge_index, alpha

if __name__ == '__main__':


    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
    device = torch.device('cuda:0')

    graph_dataset = GraphData(root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=device, )
    print(len(graph_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=graph_dataset,
                                               batch_size=1,
                                               num_workers=0,)
                                               # collate_fn=lambda x: x)
    tempo_relation = TemporalRelationGraph(256,32)
    for i, (node_feat, edge_ixs) in enumerate(train_loader):

        print(node_feat.shape)
        print(edge_ixs.shape)
        print(edge_ixs)

        tempo_res, edge_index, alpha = tempo_relation(node_feat.squeeze(0),
                                   edge_ixs.squeeze(0))
        print(tempo_res.shape)
        print(edge_index.shape)
        print(alpha.shape)
        print(edge_index)
        print(alpha)
        # for res in tempo_res:
        #     print(res)
        break
        # print(node_feat.shape)
        # for node in node_feat:
        #
        #     print(node.shape)
        # print(len(node_feat))