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
from torch_geometric.nn import GATConv, GCNConv
import numpy as np
from torch_geometric.data import DataLoader, Dataset, Data
from visdom import Visdom
from tqdm import tqdm as tqdm
import time
from evaluation import compute_perform_metrics
import networkx as nx
import torch_geometric
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
mot17_train = mot17_seqs[:5]
mot17_val = mot17_seqs[5:]


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
                             end_frame=start_frame + (self.max_frame_per_graph-2),
                             device=self.device)

        if start_frame == 1:
            num_obj_prev = 0
        else:
            num_obj_prev= len(df.loc[df['frame']<start_frame])

        graph_obj = mot_graph.construct_graph_obj_new(num_obj_prev)

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
    def __init__(self, in_channels, out_channels, heads=4):
        super(TemporalRelationGraph, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.gconv1 = GCNConv(in_channels, out_channels)
        # self.gconv2 = GCNConv(out_channels, out_channels)
        self.gat = GATConv(out_channels, out_channels, heads=heads, dropout=0.0, concat=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.linear = nn.Linear(256, 2)

    def forward(self, data):
        x = data.x
        node_feat = x
        edge_index = data.edge_index
        # graph attention or graph convolution
        # gcn_x = F.relu(self.gconv1(x, edge_index))
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
        # x = F.relu(fuse + gcn_x)
        # x = self.linear(H)
        # print(H.shape)
        x = x.view(x.shape[1], x.shape[2])
        # x = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        x = torch.mul(x[edge_index[0]], x[edge_index[1]])
        x = self.linear(x)
        # print(x.shape)
        return F.log_softmax(x)


def train(train_loader, model, criterion, optimizer, epoch, device, vis):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Accuracy', ':.4e')
    precision = AverageMeter('Precision', ':.4e')
    recall = AverageMeter('Recall', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accuracy, precision, recall],
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

        optimizer.zero_grad()
        positive_vals = batch.edge_labels.sum()
        pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals
        # loss = F.binary_cross_entropy_with_logits(output.view(-1), batch.edge_labels.view(-1).float(),
        #                                           pos_weight=pos_weight)
        loss = F.nll_loss((output.to(device)), batch.edge_labels.to(device).long(),
                                                  weight=torch.tensor([1-pos_weight,pos_weight]).to(device))
        losses.update(loss.item())

        loss.backward()
        # logs = {**compute_perform_metrics(output.cpu(), batch.cpu()), **{'loss': loss}}
        #
        # accuracy.update(logs['accuracy'])
        # precision.update(logs['precision'])
        # recall.update(logs['recall'])


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
    return all_loss,\
           # logs


def validate(val_loader, model, criterion, epoch, device, vis):
    batch_time = AverageMeter('Time', ":6.3f")
    losses = AverageMeter('Loss', ":6.3f")
    accuracy = AverageMeter('Accuracy', ':.4e')
    precision = AverageMeter('Precision', ':.4e')
    recall = AverageMeter('Recall', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accuracy, precision, recall],
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
            # loss = F.binary_cross_entropy_with_logits(output.view(-1), batch.edge_labels.view(-1).float(),
            #                                           pos_weight=pos_weight)
            loss = F.nll_loss((output.to(device)), batch.edge_labels.to(device).long(),
                              weight=torch.tensor([1 - pos_weight, pos_weight]).to(device))
            # logs = {**compute_perform_metrics(output.cpu(), batch.cpu()), **{'loss': loss}}
            losses.update(loss.item())
            # accuracy.update(logs['accuracy'])
            # precision.update(logs['precision'])
            # recall.update(logs['recall'])
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

    return all_val_loss,\
           # logs


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device(0)
    # device = torch.device('cuda:6')
    save = '/home/kevinwm99/MOT/GCN/base/'
    vis = Visdom(port=19555, env='MOT-Hamadard distance')
    graph_dataset = GraphData(root=DATA_ROOT, all_seq_name=mot17_train, datasetpara=dataset_para, device=device, )
    val_graph = GraphData(root=DATA_ROOT, all_seq_name=mot17_val, datasetpara=dataset_para, device=device, )
    print(len(graph_dataset))
    train_loader = DataLoader(dataset=graph_dataset,
                              batch_size=16,
                              num_workers=16, shuffle=True, pin_memory=False)

    val_loader = DataLoader(dataset=val_graph,
                            batch_size=16,
                            num_workers=16, shuffle=True, pin_memory=False)
    model = TemporalRelationGraph(in_channels=256, out_channels=256)
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4 )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, verbose=True)
    epochs = 50
    total_train_loss = list()
    total_val_loss = list()
    best_acc = 0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # all_loss, train_logs = train(train_loader, model, criterion, optimizer, epoch, device, vis)
        # all_val_loss, val_logs = validate(val_loader, model, criterion, epoch, device, vis)
        all_loss = train(train_loader, model, criterion, optimizer, epoch, device, vis)
        all_val_loss = validate(val_loader, model, criterion, epoch, device, vis)
        scheduler.step()
        # visualize
        # for val_batch in val_loader:
        #     break
        # data = torch_geometric.data.Data(x=val_batch.x, edge_index=val_batch.edge_labels.long())
        # g = to_networkx(val_batch)
        # nx.draw(g)
        # plt.savefig("visualize.jpg")
        # exit()
        total_train_loss.append(all_loss)
        total_val_loss.append(all_val_loss)
        # if val_logs['accuracy']>best_acc:
        #     best_acc = val_logs['accuracy']
        #     fname = '/home/kevinwm99/MOT/GCN/base/models/epoch-{}-loss-{}-acc-{}.pth'.format(epoch,
        #                                                                                      np.mean(np.array(all_loss)),
        #                                                                                      val_logs['accuracy'])
        #     torch.save(model.state_dict(), fname)
        vis.line(X=[epoch], Y=[all_loss], win='total loss', name='train ', update='append',
                 opts=dict(showlegend=True, title='total loss'))
        vis.line(X=[epoch], Y=[all_val_loss], win='total loss', name='val', update='append',
                 opts=dict(showlegend=True, title='total loss'))

        # vis.line(X=[epoch], Y=[train_logs['accuracy']], win='Accuracy', name='train ', update='append',
        #          opts=dict(showlegend=True, title='Accuracy'))
        # vis.line(X=[epoch], Y=[val_logs['accuracy']], win='Accuracy', name='val ', update='append',
        #          opts=dict(showlegend=True, title='Accuracy'))
        #
        # vis.line(X=[epoch], Y=[train_logs['recall']], win='Recall', name='train ', update='append',
        #          opts=dict(showlegend=True, title='Recall'))
        # vis.line(X=[epoch], Y=[val_logs['recall']], win='Recall', name='val ', update='append',
        #          opts=dict(showlegend=True, title='Recall'))
        #
        # vis.line(X=[epoch], Y=[train_logs['precision']], win='Precision', name='train ', update='append',
        #          opts=dict(showlegend=True, title='Precision'))
        # vis.line(X=[epoch], Y=[val_logs['precision']], win='Precision', name='val ', update='append',
        #          opts=dict(showlegend=True, title='Precision'))


    import matplotlib.pyplot as plt
    plt.plot(total_train_loss)
    plt.plot(total_val_loss)
    plt.savefig('/home/kevinwm99/MOT/GCN/base/models/loss.jpg')
