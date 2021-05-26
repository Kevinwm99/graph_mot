import os.path as osp
import pandas as pd
import pickle as pkl
import numpy as np
import torch
device = torch.device("cuda:4")
data_root = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train/MOT17-02-DPM/gt/gt.txt'
seq_path = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train/MOT17-02-DPM'
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')
# short-term dataset
if __name__ == "__main__":
    det_df = pd.read_csv(data_root)
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES

    det_df['bb_left'] -= 1 # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # VERY IMPORTANT: Filter out non Target Classes (e.g. vehicles, occluderst, etc.) (see: https://arxiv.org/abs/1603.00831)
    det_df = det_df[det_df['label'].isin([1, 2])].copy()

    det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path[:-3], f'img1/{frame_num:06}.jpg'))
    frame_num = (torch.from_numpy(det_df.frame.values)).to(device)
    detection_id = torch.from_numpy(det_df.id.values)
    frame_num =frame_num.to(device)
    unique_ids = (det_df.id.unique())

    max_frame_dist = 5
    edge_ixs = []
    len_prev_object = 0
    for id_ in unique_ids:
        frame_idx = torch.where(detection_id == id_)[0]+1
        changepoints = torch.where(frame_idx[1:] != frame_idx[:-1])[0] + 1
        changepoints = torch.cat((changepoints, torch.as_tensor([frame_idx.shape[0]]).to(changepoints.device)))
        all_det_ixs = torch.arange(frame_idx.shape[0], device=frame_idx.device)
        for start_frame_ix, end_frame_ix in zip(changepoints[:-1], changepoints[1:]):
            curr_frame_ixs = all_det_ixs[start_frame_ix: end_frame_ix]
            curr_frame_num = frame_idx[curr_frame_ixs[0]]
            curr_frame_id = detection_id[curr_frame_ixs[0]]
            past_frames_ixs = torch.where(torch.abs(frame_idx[:start_frame_ix] - curr_frame_num) <= max_frame_dist)[0]

            edge_ixs.append(torch.cartesian_prod(past_frames_ixs+len_prev_object, curr_frame_ixs+len_prev_object))
        len_prev_object +=len(frame_idx)
    #
    edge_ixs = torch.cat(edge_ixs).T

    print("source: {}".format(edge_ixs[0][:400]))
    print("destination: {}".format(edge_ixs[1][:400]))
