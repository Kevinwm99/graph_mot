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
import numpy as np
from torch.backends import cudnn
# import cupyx
import os
# import scipy
from torch import Tensor
# import cupyx.scipy.sparse.coo_matrix as coo_matrix_gpu
# device = torch.device("cuda:6")

# if torch.cuda.device_count() > 1:
#       print("We have available ", torch.cuda.device_count(), "GPUs!")
torch.manual_seed(1234)
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
                'max_detects': 400,
                'edge_feats_to_use': ['secs_time_dists', 'norm_feet_x_dists', 'norm_feet_y_dists',
                                      'bb_height_dists', 'bb_width_dists', 'emb_dist'],
                }
cnn_params = {
    'model_weights_path': '/home/kevinwm99/MOT/mot_neural_solver/output/trained_models/reid/resnet50_market_cuhk_duke.tar-232'
}
DATA_ROOT = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train'
DATA_PATH = '/home/kevinwm99/MOT/mot_neural_solver/data'
mot17_seqs = [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 5, 9, 10, 11, 13)]
mot17_train = mot17_seqs[1:5]
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


class NodeModel(nn.Module):
    def __init__(self, dropout=0.0):
        super(NodeModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=256,
                                             out_channels=256,
                                             kernel_size=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=256),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=256,
                                             out_channels=256,
                                             kernel_size=1,
                                             bias=False))
                                    # nn.BatchNorm2d(num_features=256))
                                    # nn.LeakyReLU())
        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32,
        #                                      out_channels=1,
        #                                      kernel_size=(1),
        #                                      bias=False))

    def forward(self, node_feat, edge_feat):
        aggregate = node_feat*edge_feat
        x = self.conv1(aggregate)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        # x = torch.sigmoid(x.view(1, 256))
        x = torch.sigmoid(x)
        return x


class EdgeModel(nn.Module):
    def __init__(self,):
        super(EdgeModel, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=256,
                                             out_channels=128,
                                             kernel_size=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64,
                                             out_channels=32,
                                             kernel_size=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=32,
                                             out_channels=1,
                                             kernel_size=1,
                                             bias=False))

    def forward(self, node_feat, edge_feat):
        agg_feat = node_feat*edge_feat
        x = self.conv1(agg_feat)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = torch.sigmoid(x)
        return x


class GraphNetwork(nn.Module):
    def __init__(self,
                 num_layers=5,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_node_per_graph = 500
        # for each layer
        for l in range(self.num_layers):
            # set edge to node
            edge2node_net = NodeModel()

            # set node to edge
            node2edge_net = EdgeModel()

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)

    # forward
    def forward(self, node_feat, edge_feat):
        batch_size = node_feat.size(0)
        edge_attr_list = []
        # for each layer
        for l in range(self.num_layers):
            # node update
            node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)
            edge_attr_list.append(edge_feat.view(batch_size, 1, self.num_node_per_graph, self.num_node_per_graph))


        # if tt.arg.visualization:
        #     for l in range(self.num_layers):
        #         ax = sns.heatmap(tt.nvar(edge_feat_list[l][0, 0, :, :]), xticklabels=False, yticklabels=False, linewidth=0.1,  cmap="coolwarm",  cbar=False, square=True)
        #         ax.get_figure().savefig('./visualization/edge_feat_layer{}.png'.format(l))

        return edge_attr_list


class GraphData(torch.utils.data.Dataset):
    def __init__(self, root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=None):
        super(GraphData, self).__init__()
        self.num_node_per_graph = 500
        self.max_frame_per_graph = 10
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
            # df_len = len(df)
            # if seq_name not in self.seq_index:
            #     self.seq_index[seq_name] = []
            for i in range(1, len(frames)-self.max_frame_per_graph+2):
                self.seq_index.append([seq_name, i])
                # self.seq_index[seq_name].append(i)

    def __len__(self):
        return len(self.seq_index)

    def __getitem__(self, index):
        seq_name, start_frame = self.seq_index[index]
        processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
        df, frames = processor.load_or_process_detections()
        # df_len = len(df)
        # max_frame_per_graph = 15
        fps = df.seq_info_dict['fps']
        print("Construct graph {} from frame {} to frame {}".format(seq_name, start_frame, start_frame + self.max_frame_per_graph))

        mot_graph_past = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                  start_frame=start_frame,
                                  end_frame=start_frame + (self.max_frame_per_graph-2))
        mot_graph_future = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                    start_frame=start_frame + (self.max_frame_per_graph-1),
                                    end_frame=start_frame + (self.max_frame_per_graph-1))
        mot_graph_gt = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                start_frame=start_frame,
                                end_frame=start_frame + (self.max_frame_per_graph-1))

        node_gt, _ = mot_graph_gt._load_appearance_data()
        edge_ixs_gt = mot_graph_gt._get_edge_ix_gt()
        l1, l2 = (zip(*sorted(zip(edge_ixs_gt[0].numpy(), edge_ixs_gt[1].numpy()))))
        edge_ixs_gt = (torch.tensor((l1, l2)))
        # print(edge_ixs_gt)
        # print(edge_ixs_gt - len(node_gt))
        # if (edge_ixs_gt - len(node_gt))[0] < 0:
        #     edge_ixs_gt = edge_ixs_gt.clone()
        # else:
        #     edge_ixs_gt = edge_ixs_gt - len(node_gt)
        # print(edge_ixs_gt)
        node_past, _ = mot_graph_past._load_appearance_data()  # node feature
        edge_ixs_past = mot_graph_past._get_edge_ixs()
        l1, l2 = (zip(*sorted(zip(edge_ixs_past[0].numpy(), edge_ixs_past[1].numpy()))))
        edge_ixs = (torch.tensor((l1, l2)))

        if start_frame > 1:
            past_node_df = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                    start_frame=1,
                                    end_frame=start_frame - 1)
            # print(past_node_df.graph_df)
            num_past_node = len(past_node_df.graph_df)
            edge_ixs_gt = edge_ixs_gt - num_past_node
            edge_ixs_past = edge_ixs - num_past_node
        node_fut, _ = mot_graph_future._load_appearance_data()
        # mot_graph_current_df = pd.concat([mot_graph_past.graph_df, mot_graph_future.graph_df]).reset_index(
        #     drop=True).drop(['index'], axis=1)

        edge_ixs_past = to_scipy_sparse_matrix(edge_ixs_past).toarray()
        # connect all the new nodes to the past nodes
        edge_current_coo = F.pad(torch.from_numpy(edge_ixs_past),
                                 (0, node_fut.shape[0]-1, 0, node_fut.shape[0]-1), mode='constant', value=1)
        # to [2, num edges] torch tensor
        edge_current = (from_scipy_sparse_matrix(coo_matrix(edge_current_coo.cpu().numpy()))[0])
        node_current = torch.cat((node_past, node_fut))
        print(len(node_past))
        print(len(node_fut))
        print(len(node_current))
        print(edge_current[0])
        print(edge_current)
        print(len(edge_ixs_gt[0]))
        print(dataset_para['max_detects'])
        print(edge_ixs_gt)
            # edge_current = edge_current -num_past_node

        # calculate edge attributes
        # edge_feats_dict = compute_edge_feats_dict(edge_ixs=edge_current, det_df=mot_graph_current_df,
        #                                           fps=fps,
        #                                           use_cuda=True)
        # edge_feats = [edge_feats_dict[feat_names] for feat_names in dataset_para['edge_feats_to_use'] if
        #               feat_names in edge_feats_dict]
        # edge_feats = torch.stack(edge_feats).T

        emb_dists = []
        # # divide in case out of memory
        for i in range(0, edge_current.shape[1], 50000):
            emb_dists.append(F.pairwise_distance(node_current[edge_current[0][i:i + 50000]],
                                                 node_current[edge_current[1][i:i + 50000]]).view(-1, 1))
        #
        # # print("pass")
        emb_dists = torch.cat(emb_dists, dim=0)

        # print(edge_ixs_gt)
        # emb_dists = F.pairwise_distance(node_current[edge_current[0]], node_current[edge_current[1]]).view(-1, 1)
        # print(emb_dists.shape)
        # Add embedding distances to edge features if needed
        # if 'emb_dist' in dataset_para['edge_feats_to_use']:
        #     edge_feats = torch.cat((edge_feats.to(device), emb_dists.to(device)), dim=1)
        # print("Edge features", edge_feats.shape)
        # print("Edge features", emb_dists.shape)
        # edge_attr = torch.cat((edge_feats, edge_feats))
        edge_attr = torch.cat((emb_dists, emb_dists))
        # print("Edge weight shape: {}".format(edge_attr.shape))
        # print("Edge index: {}".format(edge_current.shape))
        # print("Node features: {}".format(node_current.shape))
        # row, col = edge_current
        # print(edge_attr[row].shape, edge_attr[col].shape)
        # print(node_current[row].shape, node_current[col].shape)

        row, col = edge_current
        # for i in range(len(row)):
        #     if row[i] >=500 or col[i]>=500:
        #         print(edge_current[0][:50])
        #         print(edge_current[1][:50])
        #         print(seq_name)
        #         print(start_frame)
        neighbors_node = torch.zeros((self.num_node_per_graph, self.num_node_per_graph, 256))
        neighbors_edge = torch.zeros((self.num_node_per_graph, self.num_node_per_graph, 1))
        # neighbors_index = {}
        for i in range(len(row)):  # for all nodes
            neighbors_edge[int(row[i])][int(col[i])] = (edge_attr[int(col[i])])
            neighbors_node[int(row[i])][int(col[i])] = (node_current[int(col[i])])
            # neighbors_index[int(row[i])].append(col[i])
        node_feat = neighbors_node.view(256, self.num_node_per_graph, self.num_node_per_graph)
        edge_feat = neighbors_edge.view(1, self.num_node_per_graph, self.num_node_per_graph)
        id_label = torch.from_numpy(to_scipy_sparse_matrix(edge_ixs_gt).toarray()).argmax(dim=1).long()
        label = torch.zeros(self.num_node_per_graph)
        # make self loop for nodes that have no connection
        for i, v in enumerate(id_label):
            if id_label[i] == 0:
                id_label[i] = i
        label[0:len(id_label)] = id_label
        for i, v in enumerate(label):
            if label[i] == 0:
                label[i] = i
        # print(label.shape)
        return node_feat, edge_feat, label


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
    device = torch.device('cuda:5')
    # cudnn.benchmark = True
    # # if torch.cuda.is_available():
    # #     device = torch.device('cuda')
    graph_dataset = GraphData(root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=device,)
    print(len(graph_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=graph_dataset,
                                               batch_size=1,
                                               num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction='none')
    epochs = 10
    num_layers = 5

    # import torchvision
    # net = torchvision.models.resnet50(True).to(device)
    # net = nn.DataParallel(net)
    # net = net.to(device)
    #
    # X = torch.rand(100,3,224,224).to(device)
    # for i in range(100):
    #     x = X[i]
    #     output = net(x)
    # exit()
    # print("passss")
    graph = GraphNetwork(num_layers=num_layers).to(device)
    # graph = nn.DataParallel(graph, device_ids=[6,7,8,9])
    # graph = nn.DataParallel(graph)

    # print("pass1")
    graph = graph.to(device)


    optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3, weight_decay=0.01)
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        for i, (node_feat, edge_feat, label) in enumerate(train_loader):
            optimizer.zero_grad()
            # print(node_feat.shape)
            # print(edge_feat.shape)
            # print(label.shape)
            predictions = graph(node_feat.to(device), edge_feat.to(device))
            # print("pass2")
            # # print(predictions[0])
            #
            loss_each_layer = [criterion(prediction.squeeze(0), label.long().to(device)) for prediction in predictions]

            total_loss = []
            for l in range(num_layers-1):
                total_loss += [loss_each_layer[l].view(-1) * 0.5]
            total_loss += [loss_each_layer[-1].view(-1) * 1.0]
            total_loss = torch.mean(torch.cat(total_loss, 0))

            print("loss epoch {}".format(epoch), total_loss)
            total_loss.backward()

            optimizer.step()
            running_loss += total_loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    #######################################################################################################
                # nn.DataPar
    # for node_feat, edge_feat, label in train_loader:
    #     # print(node_feat.shape, edge_feat.shape, label.shape)
    #     # break
    #     print(label.shape)
        # print()
    # with torch.autograd.set_detect_anomaly(True):
    # #################################################################################################################################################
    # #################################################################################################################################################
    #     criterion = nn.CrossEntropyLoss(reduction='none')
    #     epochs = 10
    #     num_layers = 5
    #     num_node_per_graph = 500
    #     graph = GraphNetwork(num_layers=num_layers).to(device)
    #     optimizer = torch.optim.Adam(graph.parameters(), lr=1e-3, weight_decay=0.01)

    # for seq_name in mot17_train:
    #
    #     processor = MOTSeqProcessor(DATA_ROOT, seq_name, dataset_para, device=device)
    #     df, frames = processor.load_or_process_detections()
    #     df_len = len(df)
    #     max_frame_per_graph = 15
    #     fps = df.seq_info_dict['fps']
    #     # for i in range(1, len(frames)-max_frame_per_graph+2):
    #     #     print("Construct graph {} from frame {} to frame {}".format(seq_name,i, i+14))
    #     mot_graph_past = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
    #                               start_frame=190,
    #                               end_frame=190+13)
    #     mot_graph_future = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
    #                                 start_frame=190+14,
    #                                 end_frame=190+14)
    #     mot_graph_gt = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
    #                             start_frame=190,
    #                             end_frame=190+14)
    #
    #     node_gt,_ = mot_graph_gt._load_appearance_data()
    #     edge_ixs_gt = mot_graph_gt._get_edge_ix_gt()
    #     l1, l2 = (zip(*sorted(zip(edge_ixs_gt[0].numpy(), edge_ixs_gt[1].numpy()))))
    #     edge_ixs_gt = (torch.tensor((l1, l2)))
    #
    #     node_past, _ = mot_graph_past._load_appearance_data() # node feature
    #     edge_ixs_past = mot_graph_past._get_edge_ixs()
    #     l1, l2 = (zip(*sorted(zip(edge_ixs_past[0].numpy(), edge_ixs_past[1].numpy()))))
    #     edge_ixs = (torch.tensor((l1, l2)))
    #     print(edge_ixs_gt)
    #     # print(edge_ixs)
    #     node_fut, _ = mot_graph_future._load_appearance_data()
    #     mot_graph_current_df = pd.concat([mot_graph_past.graph_df, mot_graph_future.graph_df]).reset_index(drop=True).drop(['index'], axis=1)
    #
    #     if 190 > 1:
    #         past_node_df = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
    #                                 start_frame=1,
    #                                 end_frame=190 - 1)
    #         # print(past_node_df.graph_df)
    #         num_past_node = len(past_node_df.graph_df)
    #         edge_ixs_gt = edge_ixs_gt - num_past_node
    #         edge_ixs_past = edge_ixs - num_past_node
    #     # print(edge_ixs_gt)
    #     # print(edge_ixs_past)
    #     edge_ixs_past = to_scipy_sparse_matrix(edge_ixs_past).toarray()
    #     # edge_ixs_past = to_scipy_sparse_matrix(edge_ixs_past).toarray()
    #     # connect all the new nodes to the past nodes
    #     edge_current_coo = F.pad(torch.from_numpy(edge_ixs_past),
    #                              (0, node_fut.shape[0]-1, 0, node_fut.shape[0]-1), mode='constant', value=1)
    #     # to [2, num edges] torch tensor
    #     edge_current = (from_scipy_sparse_matrix(coo_matrix(edge_current_coo.cpu().numpy()))[0])
    #     print(edge_current[:50])
    #     print(edge_ixs_gt)
    #     print(len(edge_ixs_gt[0]))
    #     print(torch.from_numpy(to_scipy_sparse_matrix(edge_ixs_gt).toarray()))
    #     id_label = torch.from_numpy(to_scipy_sparse_matrix(edge_ixs_gt).toarray()).argmax(dim=1).long()
    #     print(id_label)
    #     print(len(id_label))
    #     label = torch.zeros(500)
    #     # make self loop for nodes that have no connection
    #     for i, v in enumerate(id_label):
    #         if id_label[i] == 0:
    #             id_label[i] = i
    #     label[0:len(id_label)] = id_label
    #     for i, v in enumerate(label):
    #         if label[i] == 0:
    #             label[i] = i
    #     node_current = torch.cat((node_past,node_fut))
    #
    #     # calculate edge attributes
    #     # edge_feats_dict = compute_edge_feats_dict(edge_ixs=edge_current, det_df=mot_graph_current_df,
    #     #                                           fps=fps,
    #     #                                           use_cuda=True)
    #     # edge_feats = [edge_feats_dict[feat_names] for feat_names in dataset_para['edge_feats_to_use'] if
    #     #               feat_names in edge_feats_dict]
    #     # edge_feats = torch.stack(edge_feats).T
    #     emb_dists = []
    #     # divide in case out of memory
    #     for i in range(0, edge_current.shape[1], 50000):
    #         emb_dists.append(F.pairwise_distance(node_current[edge_current[0][i:i + 50000]],
    #                                              node_current[edge_current[1][i:i + 50000]]).view(-1, 1))
    #
    #     emb_dists = torch.cat(emb_dists, dim=0)
    #
    #     # Add embedding distances to edge features if needed
    #     # if 'emb_dist' in dataset_para['edge_feats_to_use']:
    #     #     edge_feats = torch.cat((edge_feats.to(device), emb_dists.to(device)), dim=1)
    #     # print("Edge features", edge_feats.shape)
    #     print("Edge features", emb_dists.shape)
    #     # edge_attr = torch.cat((edge_feats, edge_feats))
    #     edge_attr = torch.cat((emb_dists, emb_dists))
    #     print("Edge weight shape: {}".format(edge_attr.shape))
    #     print("Edge index: {}".format(edge_current.shape))
    #     print("Node features: {}".format(node_current.shape))
    #     # row, col = edge_current
    #     # print(edge_attr[row].shape, edge_attr[col].shape)
    #     # print(node_current[row].shape, node_current[col].shape)
    #
    #
    #     row, col = edge_current
    #     node_feat_with_neighbors = torch.zeros((num_node_per_graph, num_node_per_graph))
    #     # for node in range(len(row)):
    #     #     node_feat_with_neighbors[row[node]][col[node]] = node_current[col[node]]
    #     neighbors_node = torch.zeros((num_node_per_graph, num_node_per_graph, 256))
    #     neighbors_edge = torch.zeros((num_node_per_graph, num_node_per_graph, 1))
    #     neighbors_index = {}
    #     for i in range(len(row)):  # for all nodes
    #         neighbors_edge[int(row[i])][int(col[i])] = (edge_attr[int(col[i])])
    #         neighbors_node[int(row[i])][int(col[i])] = (node_current[int(col[i])])
    #         # neighbors_index[int(row[i])].append(col[i])
    #     node_feat = neighbors_node.unsqueeze(0).view(1, 256, 500, 500)
    #     edge_feat = neighbors_edge.unsqueeze(0).view(1, 1, 500, 500)
    #     id_label = torch.from_numpy(to_scipy_sparse_matrix(edge_ixs_gt).toarray()).argmax(dim=1).long()
    #     label = torch.zeros(500)
    #     # make self loop for nodes that have no connection
    #     for i, v in enumerate(id_label):
    #         if id_label[i] == 0:
    #             id_label[i] = i
    #     label[0:len(id_label)] = id_label
    #     for i, v in enumerate(label):
    #         if label[i] == 0:
    #             label[i] = i
    #     break
        #         for epoch in range(epochs):
        #             predictions = graph(node_feat.to(device), edge_feat.to(device))
        #             # print(predictions[0])
        #
        #             loss_each_layer = [criterion(prediction.squeeze(0), label.unsqueeze(0).long().to(device)) for prediction in predictions]
        #
        #             total_loss = []
        #             for l in range(num_layers-1):
        #                 total_loss += [loss_each_layer[l].view(-1) * 0.5]
        #             total_loss += [loss_each_layer[-1].view(-1) * 1.0]
        #
        #             total_loss = torch.mean(torch.cat(total_loss, 0))
        #             print("loss epoch {}".format(epoch), total_loss)
        #             total_loss.backward()
        #             optimizer.step()
                # print(label.long())
                # print(prediction[0].shape)
                # print(id_label.shape)
                # print(id_label)
                # for epoch in range(epochs - 1):
                #     node_feat, edge_list = graph(node_current, edge_current, edge_attr)
                #     optimizer.zero_grad()
                #     id_label = torch.from_numpy(to_scipy_sparse_matrix(edge_ixs_gt).toarray()).argmax(dim=1).long()
                #
                #     # make self loop for nodes that have no connection
                #     for i, v in enumerate(id_label):
                #         if id_label[i] == 0:
                #             id_label[i] = i
                #
                #     loss_each_layer = [criterion(prediction.unsqueeze(0), id_label.unsqueeze(0).to(device))
                #                        for prediction in edge_list]
                #     total_loss = []
                #     for l in range(num_layers-1):
                #         total_loss += [loss_each_layer[l].view(-1) * 0.5]
                #     total_loss += [loss_each_layer[-1].view(-1) * 1.0]
                #
                #     total_loss = torch.mean(torch.cat(total_loss, 0))
                #     print("loss epoch {}".format(epoch), total_loss)
                #     total_loss.backward(retain_graph=True)
                #     optimizer.step()

            # break



#################################################################################################################################################
#################################################################################################################################################
