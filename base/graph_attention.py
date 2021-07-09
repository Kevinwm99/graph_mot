import torch
import matplotlib.pyplot as plt
import pandas as pd
from mot_graph import MOTGraph
from pack import MOTSeqProcessor
from loguru import logger
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix, to_networkx
from scipy.sparse import coo_matrix
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
import numpy as np
from torch_geometric.data import DataLoader, Dataset, Data
from visdom import Visdom
from tqdm import tqdm as tqdm
import time
from evaluation import compute_perform_metrics
import networkx as nx
import torch_geometric
from utils.graph import get_knn_mask, to_undirected_graph, to_lightweight_graph
from mot_graph import Graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from lapsolver import solve_dense
from projectors import GreedyProjector, ExactProjector
from postprocessing import Postprocessor
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
                'max_detects': None,
                'edge_feats_to_use': ['secs_time_dists', 'norm_feet_x_dists', 'norm_feet_y_dists',
                                      'bb_height_dists', 'bb_width_dists', 'emb_dist'],
                'top_k_nns': 50, # Top K-nearest neighbors (w.r.t reid score) to which a node can be  connected in the graph
                'reciprocal_k_nns': True,
                'min_track_len': 2,
                }
cnn_params = {
    'model_weights_path': '/home/kevinwm99/MOT/mot_neural_solver/output/trained_models/reid/resnet50_market_cuhk_duke.tar-232'
}
DATA_ROOT = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train'
DATA_PATH = '/home/kevinwm99/MOT/mot_neural_solver/data'
mot17_seqs = [f'MOT17-{seq_num:02}-GT' for seq_num in (2, 4, 5, 9, 10, 11, 13)]
mot17_train = mot17_seqs[:5]
mot17_val = mot17_seqs[5:]
VIDEO_COLUMNS = ['frame_path', 'frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'bb_right', 'bb_bot']
TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        self.total_num_edges = 0
        self.total_num_nodes = 0
        for seq_name in self.all_seq_name:
            processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
            df, frames = processor.load_or_process_detections()
            for start_frame in range(1, len(frames)-self.max_frame_per_graph+2):
                self.seq_index.append([seq_name, start_frame])
                # mot_graph = MOTGraph(seq_det_df=df,
                #                      seq_info_dict=df.seq_info_dict,
                #                      dataset_params=self.dataset_para,
                #                      start_frame=start_frame,
                #                      max_frame_dist=5,
                #                      end_frame=start_frame + (self.max_frame_per_graph - 2))
                # n_node, n_edge = mot_graph.get_nodes_edges()
                # self.total_num_nodes += n_node
                # self.total_num_edges += n_edge

    def __len__(self):
        return len(self.seq_index)

    def _get_full_graph_from_seq_name(self, seq_name):
        processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
        df, frames = processor.load_or_process_detections()
        mot_graph = MOTGraph(seq_det_df=df,
                             seq_info_dict=df.seq_info_dict,
                             dataset_params=self.dataset_para,
                             start_frame=1,
                             max_frame_dist=5,
                             ensure_end_is_in=True,
                             end_frame=None
                             # end_frame=start_frame + (self.max_frame_per_graph - 2)
                             )
        graph_obj = mot_graph.construct_graph_obj_new()
        return mot_graph, frames

    def __getitem__(self, index):
        seq_name, start_frame = self.seq_index[index]

        # need to fix this, load dataframe multiple times is not memory efficient
        processor = MOTSeqProcessor(self.root, seq_name, self.dataset_para, device=self.device)
        df, frames = processor.load_or_process_detections()
        self.df = df
        self.frames = frames
        # fps = df.seq_info_dict['fps']
        # print("Construct graph {} from frame {} to frame {}".format(seq_name, start_frame, start_frame + self.max_frame_per_graph))

        mot_graph = MOTGraph(seq_det_df=df,
                             seq_info_dict=df.seq_info_dict,
                             dataset_params=self.dataset_para,
                             start_frame=start_frame,
                             max_frame_dist=5,
                             end_frame=start_frame + (self.max_frame_per_graph-2))

        # if start_frame == 1:
        #     num_obj_prev = 0
        # else:
        #     num_obj_prev= len(df.loc[df['frame']<start_frame])

        graph_obj = mot_graph.construct_graph_obj_new()

        print(self.total_num_edges)
        print(self.total_num_nodes)
        print(df.frame)
        print(df.frames)
        print(len(df.frame.values))
        print(frames)
        exit()
        # node_feat, edge_ixs, labels, gt_ids = mot_graph.load_node_and_edge(num_obj_prev)

        #
        # graph_obj = Data(x=node_feat,
        #                  edge_attr=None,
        #                  edge_index=edge_ixs,
        #                  y=labels,
        #                  )
        #
        # graph_obj_gt = Data(x=node_feat,
        #                     edge_index=gt_ids)

        return graph_obj


class TemporalRelationGraph(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8):
        super(TemporalRelationGraph, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        # self.gconv1 = GCNConv(in_channels, out_channels)
        # self.gconv2 = GCNConv(out_channels, out_channels)
        self.gat = GATConv(out_channels, out_channels, heads=heads, dropout=0.0, concat=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.linear = nn.Linear(256, 128)

    def forward(self, data):
        x = data.x
        node_feat = x
        edge_index = data.edge_index
        # graph attention or graph convolution
        # x = F.relu(self.gconv1(x, edge_index))
        # x = F.relu(self.gconv2(x, edge_index))

        x = (self.gat(x, edge_index,))
        x = torch.cat([x.split(self.out_channels, dim=1)[i].unsqueeze(0) for i in range(self.heads)],
                      dim=0).unsqueeze(1)
        z = x
        # multi head relation aggregator
        x = self.global_pool(x)
        # print("pool", x)
        x = self.conv1(x)
        # print("conv: ",x)
        x = F.relu(x)
        x = F.softmax(x, dim=0)
        # print("softmax: ",x)
        fuse = torch.sum(z * x, dim=0)
        # print("fuse: ", fuse)
        # print("node feat:", node_feat)
        # print(fuse.shape)
        # fuse = self.linear(fuse.view(fuse.shape[2],shape[1]))
        x = F.relu(fuse + node_feat)
        # x = self.linear(H)
        # print(H.shape)
        x = x.view(x.shape[1], x.shape[2])
        x = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        return x


def train(train_loader, model, criterion, optimizer, epoch, device, vis):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    total_tqdm = len(train_loader)
    pbar = tqdm(total=total_tqdm, position=0, leave=True)
    all_loss = list()
    for i, (batch) in enumerate(train_loader):
        data_time.update(time.time() - end)

        batch = batch.to(device, non_blocking=True)
        # batch_gt = batch_gt.to(device, non_blocking=True)
        output = model(batch)
        # # prepare label
        # label = []
        # edge_ix_gt = to_scipy_sparse_matrix(batch_gt.edge_index).toarray()
        # edge_ix = to_scipy_sparse_matrix(batch.edge_index).toarray()
        # for j in range(len(edge_ix)):
        #     mask = (np.array(edge_ix[j], dtype=int) > 0)
        #     val = np.array(edge_ix_gt[j][mask])
        #     if val.size != 0:
        #         label.append(torch.from_numpy(val))
        # label = torch.cat(label)

        optimizer.zero_grad()
        positive_vals = batch.edge_labels.sum()
        pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals
        loss = F.binary_cross_entropy_with_logits(output.view(-1), batch.edge_labels.view(-1).float(),
                                                  pos_weight=pos_weight)
        logs = {**compute_perform_metrics(output, batch), **{'loss': loss}}
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        # if i % 20:
        #     progress.display(i)
        running_loss = loss.clone().detach().cpu().item()
        all_loss.append(running_loss)
        pbar.update()
        pbar.set_description('train running loss: %.4f' % (running_loss))
        #     running_loss = 0.0

        vis.line(X=[i + epoch * total_tqdm], Y=[running_loss], win='train running loss', name='train', update='append',
                 opts=dict(showlegend=True, title=' iter training loss'))
    all_loss = np.mean(np.array(all_loss))
    # fname = '/home/kevinwm99/MOT/GCN/base/models/epoch-{}-loss-{}.pth'.format(epoch, np.mean(np.array(all_loss)))
    # torch.save(model.state_dict(), fname)
    return all_loss, logs


def validate(val_loader, model, criterion, epoch, device, vis):
    batch_time = AverageMeter('Time', ":6.3f")
    losses = AverageMeter('Loss', ":6.3f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix="Test: "
    )
    model.eval()

    with torch.no_grad():
        end = time.time()
        total_tqdm = len(val_loader)
        pbar = tqdm(total=total_tqdm, position=0, leave=True)
        all_val_loss = list()
        for i, (batch) in enumerate(val_loader):
            batch = batch.to(device, non_blocking=True)
            output = model(batch)
            positive_vals = batch.edge_labels.sum()
            pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals
            loss = F.binary_cross_entropy_with_logits(output.view(-1), batch.edge_labels.view(-1).float(),
                                                      pos_weight=pos_weight)
            logs = {**compute_perform_metrics(output, batch), **{'loss': loss}}
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            # if i % 20:
            #     progress.display(i)
            running_loss = loss.clone().detach().cpu().item()
            all_val_loss.append(running_loss)
            pbar.update()
            pbar.set_description('val running loss: %.4f' % (running_loss))
            vis.line(X=[i + epoch * total_tqdm], Y=[running_loss], win='running val loss', name='val',
                     update='append',
                     opts=dict(showlegend=True, title=' iter val loss'))
        all_val_loss = np.mean(np.array(all_val_loss))

    return all_val_loss, logs


def predict_edges(model, subgraph):
    """
    Predicts edge values for a subgraph (i.e. batch of frames) from the entire sequence.
    Args:
        subgraph: Graph Object corresponding to a subset of frames

    Returns:
        tuple containing a torch.Tensor with the predicted value for every edge in the subgraph, and a binary mask
        indicating which edges inside the subgraph where pruned with KNN
    """
    # Prune graph edges

    knn_mask = get_knn_mask(pwise_dist=subgraph.reid_emb_dists, edge_ixs=subgraph.edge_index,
                            num_nodes=subgraph.num_nodes, top_k_nns=dataset_para['top_k_nns'],
                            use_cuda=True, reciprocal_k_nns=dataset_para['reciprocal_k_nns'],
                            symmetric_edges=True)
    subgraph.edge_index = subgraph.edge_index.T[knn_mask].T
    # subgraph.edge_attr = subgraph.edge_attr[knn_mask]
    if hasattr(subgraph, 'edge_labels'):
        subgraph.edge_labels = subgraph.edge_labels[knn_mask]

    # Predict active edges
    # if self.use_gt:  # For debugging purposes and obtaining oracle results
    # pruned_edge_preds = subgraph.edge_labels

    # else:
    with torch.no_grad():
        pruned_edge_preds = model(subgraph).view(-1)

    edge_preds = torch.zeros(knn_mask.shape[0]).to(pruned_edge_preds.device)
    edge_preds[knn_mask] = pruned_edge_preds

    # if self.eval_params['set_pruned_edges_to_inactive']:
    #     return edge_preds, torch.ones_like(knn_mask)
    #
    # else:
    return edge_preds, knn_mask


def _project_graph_model_output( subseq_graph):
    """
    Rounds MPN predictions either via Linear Programming or a greedy heuristic
    """
    # if self.eval_params['rounding_method'] == 'greedy':
    projector = GreedyProjector(subseq_graph)

    # elif self.eval_params['rounding_method'] == 'exact':
    # projector = ExactProjector(subseq_graph, solver_backend='pulp')
    #
    # else:
    #     raise RuntimeError("Rounding type for projector not understood")

    projector.project()

    subseq_graph.graph_obj = subseq_graph.graph_obj.cpu().numpy()
    subseq_graph.constr_satisf_rate = projector.constr_satisf_rate

    return subseq_graph

def _assign_ped_ids(subseq_graph):
    """
    Assigns pedestrian Ids to each detection in the sequence, by determining all connected components in the graph
    """
    # Only keep the non-zero edges and Express the result as a CSR matrix so that it can be fed to 'connected_components')
    nonzero_mask = subseq_graph.graph_obj.edge_preds == 1
    nonzero_edge_index = subseq_graph.graph_obj.edge_index.T[nonzero_mask].T
    nonzero_edges = subseq_graph.graph_obj.edge_preds[nonzero_mask].astype(int)
    graph_shape = (subseq_graph.graph_obj.num_nodes, subseq_graph.graph_obj.num_nodes)
    csr_graph = csr_matrix((nonzero_edges, (tuple(nonzero_edge_index))), shape=graph_shape)

    # Get the connected Components:
    n_components, labels = connected_components(csgraph=csr_graph, directed=False, return_labels=True)
    assert len(labels) == subseq_graph.graph_df.shape[0], "Ped Ids Label format is wrong"

    # Each Connected Component is a Ped Id. Assign those values to our DataFrame:
    final_projected_output = subseq_graph.graph_df.copy()
    final_projected_output['ped_id'] = labels
    final_projected_output = final_projected_output[VIDEO_COLUMNS + ['conf', 'detection_id']].copy()

    return final_projected_output

def _save_results_to_file( seq_df, output_file_path):
    """
    Stores the tracking result to a txt file, in MOTChallenge format.
    """
    seq_df['conf'] = 1
    seq_df['x'] = -1
    seq_df['y'] = -1
    seq_df['z'] = -1

    seq_df['bb_left'] += 1  # Indexing is 1-based in the ground truth
    seq_df['bb_top'] += 1

    final_out = seq_df[TRACKING_OUT_COLS].sort_values(by=['frame', 'ped_id'])
    final_out.to_csv(output_file_path, header=False, index=False)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device(0)
    # device = torch.device('cuda:6')
    save = '/home/kevinwm99/MOT/GCN/base/'
    vis = Visdom(port=19555, env='MOT-GCN')

    graph_dataset = GraphData(root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=device, )
    val_graph = GraphData(root=DATA_ROOT, all_seq_name=mot17_val, datasetpara=dataset_para, device=device, )
    print(len(graph_dataset))
    model = TemporalRelationGraph(in_channels=256, out_channels=256)
    model.load_state_dict(torch.load('/home/kevinwm99/MOT/GCN/base/models/epoch-44-loss-1.0580683340106094-acc-0.40821871161460876.pth'))
    model.to(device)
    model.eval()
    # exit()

    g,f = (graph_dataset._get_full_graph_from_seq_name(mot17_val[1]))

    frame_num_per_node = torch.from_numpy(g.graph_df.frame.values).to(device)
    all_frames = np.array(f)
    node_names = torch.arange(g.graph_obj.x.shape[0])

    overall_edge_preds = torch.zeros(g.graph_obj.num_edges).to(device)
    overall_num_preds = overall_edge_preds.clone()

    for eval_round, (start_frame, end_frame) in enumerate(zip(all_frames, all_frames[5 - 1:])):
        assert ((start_frame <= all_frames) & (all_frames <= end_frame)).sum() == 5
        print(eval_round)
        print(start_frame)
        print(end_frame)
        exit()
        # Create and evaluate a a subgraph corresponding to a batch of frames
        nodes_mask = (start_frame <= frame_num_per_node) & (frame_num_per_node <= end_frame)
        edges_mask = nodes_mask[g.graph_obj.edge_index[0]] & nodes_mask[
            g.graph_obj.edge_index[1]]
        # print(nodes_mask)
        # print(edges_mask)
        # print(g.graph_obj.x.shape)
        # print(g.graph_obj.edge_index.shape)
        # print((g.graph_obj.edge_index.T[edges_mask].T - node_names[nodes_mask][0]).shape)
        # print(g.graph_obj.reid_emb_dists.shape)
        subgraph = Data(x=g.graph_obj.x[nodes_mask],
                             # edge_attr=g.graph_obj.edge_attr[edges_mask],
                             reid_emb_dists=g.graph_obj.reid_emb_dists[edges_mask],
                             edge_index=g.graph_obj.edge_index.T[edges_mask].T - node_names[nodes_mask][0])

        if hasattr(g.graph_obj, 'edge_labels'):
            subgraph.edge_labels = g.graph_obj.edge_labels[edges_mask]
        subgraph = subgraph.to(device)
        # print(subgraph.reid_emb_dists)
        edge_preds, pred_mask = predict_edges(model, subgraph)
        overall_edge_preds[edges_mask] += edge_preds
        overall_num_preds[torch.where(edges_mask)[0][pred_mask]] += 1
    final_edge_preds = overall_edge_preds / overall_num_preds
    final_edge_preds[torch.isnan(final_edge_preds)] = 0
    g.graph_obj.edge_preds = final_edge_preds
    # to_undirected_graph(g, attrs_to_update=('edge_preds', 'edge_labels'))
    to_lightweight_graph(g)
    g = _project_graph_model_output(g)
    g = _assign_ped_ids(g)
    postprocess = Postprocessor(g.copy(),
                                seq_info_dict=None,
                                eval_params=dataset_para)
    seq_df = postprocess.postprocess_trajectories()
    seq_df['frame']+=1
    print(seq_df)
    # _save_results_to_file(seq_df, '/home/kevinwm99/MOT/GCN/base/result/MOT17-02-GT.txt')
    # _save_results_to_file(seq_df, '/home/kevinwm99/MOT/GCN/base/result/MOT17-05-GT.txt')
    # _save_results_to_file(seq_df, '/home/kevinwm99/MOT/GCN/base/result/MOT17-09-GT.txt')
    # _save_results_to_file(seq_df, '/home/kevinwm99/MOT/GCN/base/result/MOT17-10-GT.txt')
    # _save_results_to_file(seq_df, '/home/kevinwm99/MOT/GCN/base/result/MOT17-11-GT.txt')
    _save_results_to_file(seq_df, '/home/kevinwm99/MOT/GCN/base/result/MOT17-13-GT.txt')
    exit()


    ################################################################################################################
    ################################################################################################################
    # train_loader = DataLoader(dataset=graph_dataset,
    #                           batch_size=1,
    #                           num_workers=0, shuffle=False, pin_memory=True)
    #
    # val_loader = DataLoader(dataset=val_graph,
    #                         batch_size=32,
    #                         num_workers=32, shuffle=True, pin_memory=True)
    # model = TemporalRelationGraph(in_channels=256, out_channels=256)
    # model = model.to(device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4 )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7, verbose=True)
    # epochs = 50
    # total_train_loss = list()
    # total_val_loss = list()
    # best_acc = 0
    # for epoch in range(epochs):
    #     print('Epoch {}/{}'.format(epoch, epochs - 1))
    #     print('-' * 10)
    #
    #     all_loss, train_logs = train(train_loader, model, criterion, optimizer, epoch, device, vis)
    #     all_val_loss, val_logs = validate(val_loader, model, criterion, epoch, device, vis)
    #     scheduler.step()
    #
    #     total_train_loss.append(all_loss)
    #     total_val_loss.append(all_val_loss)
    #     if val_logs['accuracy']>best_acc:
    #         best_acc = val_logs['accuracy']
    #         fname = '/home/kevinwm99/MOT/GCN/base/models/epoch-{}-loss-{}-acc-{}.pth'.format(epoch,
    #                                                                                          np.mean(np.array(all_loss)),
    #                                                                                          val_logs['accuracy'])
    #         torch.save(model.state_dict(), fname)
    #     vis.line(X=[epoch], Y=[all_loss], win='total loss', name='train ', update='append',
    #              opts=dict(showlegend=True, title='total loss'))
    #     vis.line(X=[epoch], Y=[all_val_loss], win='total loss', name='val', update='append',
    #              opts=dict(showlegend=True, title='total loss'))
    #
    #     vis.line(X=[epoch], Y=[train_logs['accuracy']], win='Accuracy', name='train ', update='append',
    #              opts=dict(showlegend=True, title='Accuracy'))
    #     vis.line(X=[epoch], Y=[val_logs['accuracy']], win='Accuracy', name='val ', update='append',
    #              opts=dict(showlegend=True, title='Accuracy'))
    #
    #     vis.line(X=[epoch], Y=[train_logs['recall']], win='Recall', name='train ', update='append',
    #              opts=dict(showlegend=True, title='Recall'))
    #     vis.line(X=[epoch], Y=[val_logs['recall']], win='Recall', name='val ', update='append',
    #              opts=dict(showlegend=True, title='Recall'))
    #
    #     vis.line(X=[epoch], Y=[train_logs['precision']], win='Precision', name='train ', update='append',
    #              opts=dict(showlegend=True, title='Precision'))
    #     vis.line(X=[epoch], Y=[val_logs['precision']], win='Precision', name='val ', update='append',
    #              opts=dict(showlegend=True, title='Precision'))
    #
    #
    # import matplotlib.pyplot as plt
    # plt.plot(total_train_loss)
    # plt.plot(total_val_loss)
    # plt.savefig('/home/kevinwm99/MOT/GCN/base/models/loss.jpg')