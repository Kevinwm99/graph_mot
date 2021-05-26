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
from datasets.dataset.jde import DetDataset, JointDataset
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat

from opts import opts
from torchvision.transforms import transforms as T
import pandas as pd
import pickle as pkl



def main(opt):




    data_cfg = opt.data_cfg
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()

    opt.gpus = [4]

    trainset_paths = data_cfg_dict['train']
    dataset_root = data_cfg_dict['root']
    # valset_path = data_cfg_dict['val']

    transforms = T.Compose([T.ToTensor()])
    Dataset = get_dataset(opt.dataset, opt.task)
    dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model=model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            # GET EMBEDDING PATH FROM DATALOADER
            embed = {}
            embed_path = [b[:-3]+"pt" for b in batch['img_path']]
            outputs = model(batch['input'].to(device))
            id_head = _tranpose_and_gather_feat(outputs[0]['id'], batch['ind'].to(device))
            id_head = id_head[batch['reg_mask'].to(device) > 0].contiguous()
            id_target = batch['ids'].to(device)[batch['reg_mask'].to(device) > 0]
            id_head=id_head.cpu()
            id_target=id_target.cpu()
            for id_, feat in zip(id_target,id_head):
                embed[id_]=feat

            for path in (embed_path):

                torch.save(embed,path)

    print("finish store embedding")
if __name__ == '__main__':
    opt = opts().init()
    main(opt)