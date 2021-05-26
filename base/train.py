import torch

from create_dataloader import MOTGraph
from pack import MOTSeqProcessor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
from torch_geometric.utils import to_scipy_sparse_matrix

from non_local_embedded_gaussian import NONLocalBlock2D
from generator import GCNStack, Generator
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix

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
                'max_detects': 500,
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
if __name__ == '__main__':

    epochs = 100
    model = Generator()
    model = model.to(device)

    model.apply(weights_init_uniform)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.01, momentum=0.9)
    criterion = torch.nn.MSELoss()
    # print(model(torch.randn(1,1,500,256).to(device)).shape)
    # summary(model, torch.randn((1,1,500,256)).to(device),print_summary=True)
    # model.summary((1,1,500,256))
    #################################################################################################################################################
    #################################################################################################################################################
    # feat = torch.randn(1,1,500,256).to(device)
    # output = model(feat)
    # mask = torch.zeros_like(output)
    # mask[output>=0.5]=1
    # sparse = (coo_matrix(mask.cpu()))
    # sparse = from_scipy_sparse_matrix(sparse)
    # print(sparse[0].shape)
    # data = Data(feat.squeeze(0).squeeze(0),sparse[0].to(device))
    # gcn = GCNStack()
    # gcn = gcn.to(device)
    # print(gcn(data))
    # print(data.data.shape)
    # print(data.edge_ix.shape)

    # summary(model, (1, 500, 256))
    # print(model.parameters())
    # summary(model,(1,500,256))
    #################################################################################################################################################
    #################################################################################################################################################
    max_loss = 1e7
    accum_loss = []
    for epoch in range(1, epochs):
        #     #
        loss_epoch =[]
        for seq_name in mot17_train:

            processor = MOTSeqProcessor(DATA_ROOT, seq_name, dataset_para, device=device)
            df = processor.load_or_process_detections()
            mot_graph = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                 start_frame=1)
            node, _ = mot_graph._load_appearance_data()
            edge_index_gt = mot_graph._get_edge_ix_gt()
            l1, l2 = (zip(*sorted(zip(edge_index_gt[0].numpy(), edge_index_gt[1].numpy()))))
            edge_index_gt = (torch.tensor((l1, l2)))
            # edge_index_gt = to_scipy_sparse_matrix(edge_index_gt).toarray()
            # print(edge_index_gt)
            # break
            # node = (node[mot_graph.graph_df.sort_values(by=['id', 'frame']).index])
            if len(node) < 500:
                node = torch.cat((node.to(device), torch.zeros((500 - len(node), 256)).to(device)))
            else:
                node = node[:500]
            node = node.unsqueeze(0).unsqueeze(0).to(device)
            sparse_matrix = (to_scipy_sparse_matrix(edge_index_gt))
            labels = sparse_matrix.toarray()
            labels = torch.from_numpy(labels).double().float().to(device)
            if len(labels) < 500:
                pad = nn.ZeroPad2d((0, 500 - len(labels), 0, 500 - len(labels)))
                labels = pad(labels)
            else:
                labels = labels[:500]

            #
            output = model(node).squeeze(0)
            # print(output.shape)
            # mask = torch.zeros_like(output.squeeze(0))
            # print(mask.shape)
            # mask[output.squeeze(0) >= 0.5] = 1  # apply threshold
            # mask = torch.clone(mask).requires_grad_(True)
            # print(torch.max(mask))
            optimizer.zero_grad()
            loss = criterion(output, labels)

            loss.backward()
            loss_epoch.append(loss.detach().item())
            # loss_epoch.append(loss)
            optimizer.step()

            if epoch % 10 == 0 and loss < max_loss:
                max_loss = loss
                print("loss: {}".format(loss))
                fname = '/home/kevinwm99/MOT/GCN/base/models/seq-{}-epoch-{}-loss-{}.pth'.format(seq_name, epoch, loss)
                torch.save(model.state_dict(), fname)
        accum_loss.append(sum(loss_epoch)/len(loss_epoch))

    import matplotlib.pyplot as plt

    plt.plot(accum_loss)
    plt.savefig("/home/kevinwm99/MOT/GCN/base/loss.png")

#################################################################################################################################################
#################################################################################################################################################
