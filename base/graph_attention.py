import torch
import matplotlib.pyplot as plt
import pandas as pd
from mot_graph import MOTGraph
from pack import MOTSeqProcessor
from loguru import logger
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.nn import GATConv
import numpy as np
from torch_geometric.data import DataLoader, Dataset, Data
from visdom import Visdom
from tqdm import tqdm as tqdm
# step = 0
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
mot17_train = mot17_seqs[2:5]
mot17_val = mot17_seqs[5:]


class GraphData(torch.utils.data.Dataset):
    def __init__(self, root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=None):
        super(GraphData, self).__init__()
        self.num_node_per_graph = 250
        self.max_frame_per_graph = 5
        self.all_seq_name = all_seq_name
        self.dataset_para = datasetpara
        self.device = device
        self.root = root
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

    def __getitem__(self, index):
        seq_name, start_frame = self.seq_index[index]
        processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
        df, frames = processor.load_or_process_detections()
        # fps = df.seq_info_dict['fps']
        # print("Construct graph {} from frame {} to frame {}".format(seq_name, start_frame, start_frame + self.max_frame_per_graph))

        mot_graph = MOTGraph(seq_det_df=df,
                             seq_info_dict=df.seq_info_dict,
                             dataset_params=dataset_para,
                             start_frame=start_frame,
                             max_frame_dist=5,
                             end_frame=start_frame + (self.max_frame_per_graph-2))

        node_feat, edge_ixs, labels, gt_ids = mot_graph.load_node_and_edge()
        print("gt id: ", gt_ids)
        print(gt_ids.shape)
        # print(labels)
        # print(labels.shape)
        # exit()
        graph_obj = Data(x=node_feat,
                         edge_attr=None,
                         edge_index=edge_ixs,
                         y=labels,
                         )
        graph_obj_gt = Data(x=node_feat,
                            edge_index=gt_ids)
        # pad_node = node_feat.size(0)
        # node_feat = F.pad(node_feat, [0, 0, 0, self.num_node_per_graph-pad_node],
        #                   mode='constant', value=0)
        # indices = edge_ixs
        # edge_ixs = to_scipy_sparse_matrix(edge_ixs).toarray()
        # edge_ixs = F.pad(torch.from_numpy(edge_ixs),
        #                          [0, self.num_node_per_graph-pad_node, 0,
        #                           self.num_node_per_graph-pad_node], mode='constant', value=0)
        # edge_ixs_coo = coo_matrix(edge_ixs.numpy())
        # values = edge_ixs_coo.data
        # # indices = np.vstack((coo.row, coo.col))
        #
        # i = torch.LongTensor(indices)
        # v = torch.FloatTensor(values)
        # shape = edge_ixs_coo.shape
        #
        # edge_ixs = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        # # print(edge_ix.shape)
        # labels +=1
        # labels = np.pad(labels, (0, self.num_node_per_graph-pad_node))

        # return node_feat, edge_ixs, labels
        # print(node_feat.shape)
        return graph_obj, graph_obj_gt


class TemporalRelationGraph(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8):
        super(TemporalRelationGraph, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=0.0, concat=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.linear = nn.Linear(256, 128)

    def forward(self, data):
        x = data.x
        node_feat = x
        edge_index = data.edge_index
        # graph attention or graph convolution
        x = (self.gat(x, edge_index,))
        x = torch.cat([x.split(self.out_channels, dim=1)[i].unsqueeze(0) for i in range(self.heads)],
                      dim=0).unsqueeze(1)
        z = x
        # multi head relation aggregator
        x = self.global_pool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.softmax(x, dim=0)
        fuse = torch.sum(z * x, dim=0)
        H = F.relu(fuse + node_feat)
        x = self.linear(H)
        print(x.shape)
        exit()
        H = H.view(H.shape[1], H.shape[2])
        print(H.shape)
        # exit()
        print(H[edge_index[0]].shape)
        # exit()
        hadamard_dist = torch.mul(H[edge_index[0]], H[edge_index[1]])
        print(hadamard_dist)
        print(hadamard_dist.shape)
        print(H.shape)
        print(edge_index)
        print(edge_index.shape)
        exit()

        return x


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
    device = torch.device('cuda:6')
    save = '/home/kevinwm99/MOT/GCN/base/'
    # vis = Visdom(port=19555, env='test')
    graph_dataset = GraphData(root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=device, )
    val_graph = GraphData(root=DATA_ROOT, all_seq_name=mot17_val, datasetpara=dataset_para, device=device, )
    print(len(graph_dataset))
    train_loader = DataLoader(dataset=graph_dataset,
                                               batch_size=4,
                                               num_workers=0)
                                               # drop_last = True tim hieu cai nay xem, nhieu khi bi anh huong nhieu
                                               # collate_fn=lambda x: x)

    val_loader = torch.utils.data.DataLoader(dataset=val_graph,
                                               batch_size=1,
                                               num_workers=0)
    model = TemporalRelationGraph(in_channels=256, out_channels=256)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20

    for i, (batch, batch_gt) in enumerate(train_loader):
        # print(batch.x)
        # print(batch.x.shape)
        # print(batch.y)
        # print(batch.edge_index)
        batch = batch.to(device)
        batch_gt = batch_gt.to(device)
        out = model(batch)
        # exit()
    # total_loss = 0.0
    total_train_loss = list()
    total_val_loss = list()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        total_tqdm = len(train_loader)
        pbar = tqdm(total=total_tqdm, position=0, leave=True)
        all_loss = list()
        for i, (node_feat, edge_ixs, labels) in enumerate(train_loader):
            tempo_res = model(node_feat.squeeze(0).to(device), edge_ixs.squeeze(0).to(device))
            optimizer.zero_grad()
            loss = criterion(tempo_res, labels.long().to(device))
            loss.backward()
            optimizer.step()
            running_loss = loss.clone().detach().cpu().item()
            all_loss.append(running_loss)
            pbar.update()
            pbar.set_description('running loss: %.4f'%(running_loss))
            #     running_loss = 0.0

            vis.line(X=[i+ epoch*total_tqdm], Y=[running_loss], win='train running loss', name='train', update='append',
                     opts=dict(showlegend=True, title=' iter training loss'))
        total_train_loss.append(np.mean(np.array(all_loss)))

        fname = '/home/kevinwm99/MOT/GCN/base/models/epoch-{}-loss-{}.pth'.format(epoch, np.mean(np.array(all_loss)))
        torch.save(model.state_dict(), fname)

        with torch.no_grad():
            total_tqdm = len(val_loader)
            pbar = tqdm(total=total_tqdm, position=0, leave=True)
            all_loss_val = list()
            for i, (node_feat, edge_ixs, labels) in enumerate(val_loader):
                tempo_res = model(node_feat.squeeze(0).to(device), edge_ixs.squeeze(0).to(device))
                loss = criterion(tempo_res, labels.long().to(device))
                running_loss = loss.clone().detach().cpu().item()
                all_loss_val.append(running_loss)
                pbar.update()
                pbar.set_description('val loss: %.4f' % (running_loss))
                vis.line(X=[i + epoch*total_tqdm], Y=[running_loss], win='running val loss', name='val',
                         update='append',
                         opts=dict(showlegend=True, title=' iter val loss'))
            val_loss = np.mean(np.array(all_loss_val))
        total_val_loss.append(val_loss)
        vis.line(X=[epoch], Y=[np.mean(np.array(all_loss))], win='total loss', name='train ', update='append',
                 opts=dict(showlegend=True, title='total loss'))
        vis.line(X=[epoch], Y=[np.mean(np.array(all_loss_val))], win='total loss', name='val', update='append',
                 opts=dict(showlegend=True, title='total loss'))
    import matplotlib.pyplot as plt
    plt.plot(total_train_loss)
    plt.plot(total_val_loss)
    plt.savefig('/home/kevinwm99/MOT/GCN/base/models/loss.jpg')
