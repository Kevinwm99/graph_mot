from __future__ import absolute_import
import init_paths
import numpy as np
import torch
import random
import math
import time
import json
import os
device = torch.device("cuda:4")
torch.cuda.set_device(4)
from torch.utils.data.sampler import Sampler
from datasets.dataset.jde import DetDataset,JointDataset
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat

from opts import opts
from torchvision.transforms import transforms as T
import pandas as pd
import pickle as pkl

def cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to length 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a)
        b = np.asarray(b) / np.linalg.norm(b)
    return 1. - np.dot(a, b.T)


def main(opt):
    data_cfg = opt.data_cfg
    
    opt.gpus=[4]
    print(opt.gpus)
    f = open(data_cfg)
    data_cfg_dict = json.load(f)

    nC = 1
    trainset_paths = data_cfg_dict['train']
    dataset_root = data_cfg_dict['root']
    f.close()
    transforms = T.Compose([T.ToTensor()])
    Dataset = get_dataset(opt.dataset, opt.task)
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), transforms=transforms)
    dataset2 = Dataset(opt, dataset_root, trainset_paths, (1088, 608), transforms=transforms, next_idx=True)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    train_loader2 = torch.utils.data.DataLoader(
        dataset2,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(11111111111111)
    print(opt.gpus[0])
    print(11111111111111)
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    # model = torch.nn.DataParallel(model)
    model = model.to(opt.device)

    outputs, mAPs, mR, mP, TP, confidence, pred_class, target_class, jdict = \
        [], [], [], [], [], [], [], [], []
    data ={'ids':[],
           'bbox':[],
           'reid':[],
           'fid':[],
           'hm':[],
           'wh':[],
           'offset':[]}
    edge_index=[]
    edge_index_sequence=[[],[]]
    edge_weight = []
    edge_weight_sequence=[]
    flag=False
    for i, batch in enumerate(zip(train_loader,train_loader2)):
        if i==0 or i==len(train_loader)-1:
            continue

        batch_next = batch[1]
        batch = batch[0]
        # print(batch['img_path'][0][60:72])
        # print(batch_next['img_path'][0][60:72])
        if batch['img_path'][0][60:72]==batch_next['img_path'][0][60:72]:
            # print("ok")

            # same sequence
            # frame
            outputs2 = model(batch_next['input'].cuda())
            id_head2 = _tranpose_and_gather_feat(outputs2[0]['id'], batch_next['ind'].cuda())
            id_head2 = id_head2[batch_next['reg_mask'].cuda() > 0].contiguous()
            id_target2 = batch_next['ids'].cuda()[batch_next['reg_mask'].cuda() > 0]
            id_box2 = batch_next['bbox'].cuda()[batch_next['reg_mask'].cuda() > 0]

            # next frame
            outputs = model(batch['input'].cuda())
            id_head = _tranpose_and_gather_feat(outputs[0]['id'], batch['ind'].cuda())
            id_head = id_head[batch['reg_mask'].cuda() > 0].contiguous()
            id_target = batch['ids'].cuda()[batch['reg_mask'].cuda() > 0]
            id_box = batch['bbox'].cuda()[batch['reg_mask'].cuda() > 0]

            node_id_batch = i
            node_id_batch_next = i+1
            weights =[]
            for j,id_ in enumerate(zip(id_target,id_target2)):
                if id_[0]==id_[1]:
                    # print(id_head2[j].detach().cpu())
                    edge_index_sequence[0].append(node_id_batch)
                    edge_index_sequence[1].append(node_id_batch_next)
                    weights.append(cosine_distance(id_head2[j].detach().cpu(),id_head[j].detach().cpu()))
            edge_weight_sequence.append(weights)
            print(edge_index_sequence)
            if flag:
                flag = False
                print(batch['img_path'])
                print(batch_next['img_path'])
                edge_weight.append(edge_weight_sequence)
                edge_index.append(edge_index_sequence)



        else:
            flag = True
            print("switch sequence")
            print("reset edge index sequence")
            edge_index_sequence=[[],[]]
            print(batch['img_path'])
            print(batch_next['img_path'])
            # break
        # frame
        # outputs2 = model(batch_next['input'].cuda())
        # id_head2 = _tranpose_and_gather_feat(outputs2[0]['id'], batch_next['ind'].cuda())
        # id_head2 = id_head2[batch_next['reg_mask'].cuda() > 0].contiguous()
        # id_target2 = batch_next['ids'].cuda()[batch_next['reg_mask'].cuda() > 0]
        # id_box2 = batch_next['bbox'].cuda()[batch_next['reg_mask'].cuda() > 0]
        #
        # # next frame
        # outputs = model(batch['input'].cuda())
        # id_head = _tranpose_and_gather_feat(outputs[0]['id'], batch['ind'].cuda())
        # id_head = id_head[batch['reg_mask'].cuda() > 0].contiguous()
        # id_target = batch['ids'].cuda()[batch['reg_mask'].cuda() > 0]
        # id_box = batch['bbox'].cuda()[batch['reg_mask'].cuda() > 0]

        # for id in zip(id_target,id_target2):
        #     if id[0]==id[1]:
        #
        # weight = cosine_distance(id_head, id_head2)
        #
        # print(batch['input'].shape)
        # print(id_head)
        # print(id_head2)

        # print(batch['input'][0].cpu().permute(2, 1, 0).numpy().shape)
        # print(len(train_loader))

        # print(batch_next['img_path'])
        # break
        # if i ==len(train_loader)-1:
        #     print(batch['img_path'])
        #     print(batch_next['img_path'])
        # if i==len(train_loader):
        #     plt.imshow(batch['input'][0].cpu().permute(1,2,0))
        #     plt.savefig("/home/kevinwm99/MOT/check1.png")
        #     plt.imshow(batch_next['input'][0].cpu().permute(1,2,0))
        #     plt.savefig("/home/kevinwm99/MOT/check2.png")
        # break

        #
        # assert id_box.shape[0]==id_head.shape[0]


        # batch_1 = next(iter(train_loader))
        # output_next = model(batch_1['input'].cuda())
        # id_head_1 = _tranpose_and_gather_feat(output_next[0]['id'], batch_1['ind'].cuda())
        # id_head_1 = id_head_1[batch_1['reg_mask'].cuda() > 0].contiguous()
        # #
        # print(id_head_1)
        # print(outputs[0]['hm'].shape) # heatmap
        # print(outputs[0]['wh'].shape) # width height
        # print(outputs[0]['id'].shape) # id features
        # print(outputs[0]['reg'].shape) # offset
        # print(id_head)
        # print(id_head.shape)
        # print(id_target)
        # print(id_target.shape)
        # print(id_box.shape)
        # print(data)

        # if i == 1:
        #     break
        # break
        # print(i)
    print(edge_weight)
    print(edge_index)
    # pkl.dump(data,"/home/kevinwm99/MOT/FairMOT/data.pkl")
if __name__ == '__main__':

    opt = opts().init()
    main(opt)
