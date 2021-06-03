import os
import os.path as osp
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import configparser
from torch.utils.data import DataLoader
from models.resnet import resnet50_fc256, load_pretrained_weights
from utils.rgb import BoundingBoxDataset
from utils.iou import iou
from lapsolver import solve_dense

device = torch.device("cuda:6")

DATA_PATH = '/home/kevinwm99/MOT/mot_neural_solver/data'
dataset_para = {'det_file_name': 'frcnn_prepr_det',
                'node_embeddings_dir': 'resnet50_conv',
                'reid_embeddings_dir': 'resnet50_w_fc256',
                'img_batch_size': 5000,  # 6GBytes
                'gt_assign_min_iou': 0.5
                }

cnn_params = {
    # arch: 'resnet50',
    'model_weights_path': '/home/kevinwm99/MOT/mot_neural_solver/output/trained_models/reid/resnet50_market_cuhk_duke.tar-232'
}
MOV_CAMERA_DICT = {'MOT17-02-GT': False,
                   'MOT17-02-SDP': False,
                   'MOT17-02-FRCNN': False,
                   'MOT17-02-DPM': False,

                   'MOT17-04-GT': False,
                   'MOT17-04-SDP': False,
                   'MOT17-04-FRCNN': False,
                   'MOT17-04-DPM': False,

                   'MOT17-05-GT': True,
                   'MOT17-05-SDP': True,
                   'MOT17-05-FRCNN': True,
                   'MOT17-05-DPM': True,

                   'MOT17-09-GT': False,
                   'MOT17-09-SDP': False,
                   'MOT17-09-FRCNN': False,
                   'MOT17-09-DPM': False,

                   'MOT17-10-GT': True,
                   'MOT17-10-SDP': True,
                   'MOT17-10-FRCNN': True,
                   'MOT17-10-DPM': True,

                   'MOT17-11-GT': True,
                   'MOT17-11-SDP': True,
                   'MOT17-11-FRCNN': True,
                   'MOT17-11-DPM': True,

                   'MOT17-13-GT': True,
                   'MOT17-13-SDP': True,
                   'MOT17-13-FRCNN': True,
                   'MOT17-13-DPM': True,

                   'MOT17-14-SDP': True,
                   'MOT17-14-FRCNN': True,
                   'MOT17-14-DPM': True,

                   'MOT17-12-SDP': True,
                   'MOT17-12-FRCNN': True,
                   'MOT17-12-DPM': True,

                   'MOT17-08-SDP': False,
                   'MOT17-08-FRCNN': False,
                   'MOT17-08-DPM': False,

                   'MOT17-07-SDP': True,
                   'MOT17-07-FRCNN': True,
                   'MOT17-07-DPM': True,

                   'MOT17-06-SDP': True,
                   'MOT17-06-FRCNN': True,
                   'MOT17-06-DPM': True,

                   'MOT17-03-SDP': False,
                   'MOT17-03-FRCNN': False,
                   'MOT17-03-DPM': False,

                   'MOT17-01-SDP': False,
                   'MOT17-01-FRCNN': False,
                   'MOT17-01-DPM': False
                   }
_SEQ_TYPES = {}
# Determines whether boxes are allowed to have some area outside the image (all GT annotations in MOT15 are inside img
# hence we crop its detections to also be inside it)
_ENSURE_BOX_IN_FRAME = {'MOT': False,
                        'MOT_gt': False,
                        'MOT15': True,
                        'MOT15_gt': False}

# MOT17 Sequences
mot17_seqs = [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in
              ('DPM', 'SDP', 'FRCNN', 'GT')]
# mot17_seqs += [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in ('DPM', 'SDP', 'FRCNN')]
for seq_name in mot17_seqs:
    if 'GT' in seq_name:
        _SEQ_TYPES[seq_name] = 'MOT_gt'

    else:
        _SEQ_TYPES[seq_name] = 'MOT'

# We now map each sequence name to a sequence type in _SEQ_TYPES


DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'label', 'vis')


def _add_frame_path_mot17(det_df, seq_name, data_root_path):
    # Add each image's path from  MOT17Det data dir
    seq_name_wo_dets = '-'.join(seq_name.split('-')[:-1])
    det_seq_path = osp.join(data_root_path.replace('Labels', 'Det'), seq_name_wo_dets)
    add_frame_path = lambda frame_num: osp.join(det_seq_path, det_seq_path, f'img1/{frame_num:06}.jpg')
    det_df['frame_path'] = det_df['frame'].apply(add_frame_path)


def _build_scene_info_dict_mot17(seq_name, data_root_path, dataset_params):
    info_file_path = osp.join(data_root_path, seq_name, 'seqinfo.ini')
    cp = configparser.ConfigParser()
    cp.read(info_file_path)

    seq_info_dict = {'seq': seq_name,
                     'seq_path': osp.join(data_root_path, seq_name),
                     'det_file_name': dataset_params['det_file_name'],

                     'frame_height': int(cp.get('Sequence', 'imHeight')),
                     'frame_width': int(cp.get('Sequence', 'imWidth')),

                     'seq_len': int(cp.get('Sequence', 'seqLength')),
                     'fps': int(cp.get('Sequence', 'frameRate')),
                     'mov_camera': MOV_CAMERA_DICT[seq_name],

                     'has_gt': osp.exists(osp.join(data_root_path, seq_name, 'gt'))}
    return seq_info_dict


def get_mot_det_df(seq_name, data_root_path, dataset_params):
    seq_path = osp.join(data_root_path, seq_name)
    detections_file_path = osp.join(seq_path, f"det/{dataset_params['det_file_name']}.txt")
    det_df = pd.read_csv(detections_file_path, header=None)

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES)]]
    det_df.columns = DET_COL_NAMES

    det_df['bb_left'] -= 1  # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # If id already contains an ID assignment (e.g. using tracktor output), keep it
    if len(det_df['id'].unique()) > 1:
        det_df['tracktor_id'] = det_df['id']

    # Add each image's path (in MOT17Det data dir)
    if 'MOT17' in seq_name:
        _add_frame_path_mot17(det_df, seq_name, data_root_path)

    else:
        det_df['frame_path'] = det_df['frame'].apply(lambda frame_num: osp.join(seq_path, f'img1/{frame_num:06}.jpg'))

    assert osp.exists(det_df['frame_path'].iloc[0])

    seq_info_dict = _build_scene_info_dict_mot17(seq_name, data_root_path, dataset_params)
    seq_info_dict['is_gt'] = False
    if seq_info_dict['has_gt']:  # Return the corresponding ground truth, if available, for the ground truth assignment
        gt_file_path = osp.join(seq_path, f"gt/gt.txt")
        gt_df = pd.read_csv(gt_file_path, header=None)
        gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
        gt_df.columns = GT_COL_NAMES
        gt_df['bb_left'] -= 1  # Coordinates are 1 based
        gt_df['bb_top'] -= 1
        gt_df = gt_df[gt_df['label'].isin([1, 2, 7, 8, 12])].copy()  # Classes 7, 8, 12 are 'ambiguous' and tracking
        # them is not penalized, hence we keep them for the
        # GT Assignment
        # See https://arxiv.org/pdf/1603.00831.pdf
        gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
        gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

        # Store the gt file in the common evaluation path
        gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
        os.makedirs(gt_to_eval_path, exist_ok=True)
        shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    else:
        gt_df = None

    return det_df, seq_info_dict, gt_df


def get_mot_det_df_from_gt(seq_name, data_root_path, dataset_params):
    # Create a dir to store Ground truth data in case it does not exist yet
    seq_path = osp.join(data_root_path, seq_name)
    if not osp.exists(seq_path):
        os.mkdir(seq_path)

        # Copy ground truth and seq info from a seq that has this ground truth.
        if 'MOT17' in seq_name:  # For MOT17 we use e.g. the seq with DPM detections (any will do)
            src_seq_path = osp.join(data_root_path, seq_name[:-2] + 'DPM')

        else:  # Otherwise just use the actual sequence
            src_seq_path = osp.join(data_root_path, seq_name[:-3])

        shutil.copy(osp.join(src_seq_path, 'seqinfo.ini'), osp.join(seq_path, 'seqinfo.ini'))
        shutil.copytree(osp.join(src_seq_path, 'gt'), osp.join(seq_path, 'gt'))

    detections_file_path = osp.join(data_root_path, seq_name, f"gt/gt.txt")
    det_df = pd.read_csv(detections_file_path, header=None)

    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
    det_df.columns = GT_COL_NAMES

    det_df['bb_left'] -= 1  # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # VERY IMPORTANT: Filter out non Target Classes (e.g. vehicles, occluderst, etc.) (see: https://arxiv.org/abs/1603.00831)
    det_df = det_df[det_df['label'].isin([1, 2])].copy()

    if 'MOT17' in seq_name:
        _add_frame_path_mot17(det_df, seq_name, data_root_path)

    else:
        det_df['frame_path'] = det_df['frame'].apply(
            lambda frame_num: osp.join(seq_path[:-3], f'img1/{frame_num:06}.jpg'))
    assert osp.exists(det_df['frame_path'].iloc[0])

    seq_info_dict = _build_scene_info_dict_mot17(seq_name, data_root_path, dataset_params)

    # Correct the detections file name to contain the 'gt' as name
    seq_info_dict['det_file_name'] = 'gt'
    seq_info_dict['is_gt'] = True

    # Store the gt file in the common evaluation path
    gt_file_path = osp.join(seq_path, f"gt/gt.txt")
    gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
    os.makedirs(gt_to_eval_path, exist_ok=True)
    shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    return det_df, seq_info_dict, None


_SEQ_TYPE_DETS_DF_LOADER = {'MOT': get_mot_det_df,
                            'MOT_gt': get_mot_det_df_from_gt, }


# 'MOT15': get_mot15_det_df,
# 'MOT15_gt': get_mot15_det_df_from_gt}

class DataFrameWSeqInfo(pd.DataFrame):
    """
    Class used to store each sequences's processed detections as a DataFrame. We just add a metadata attribute to
    pandas DataFrames it so that sequence metainfo such as fps, etc. can be stored in the attribute 'seq_info_dict'.
    This attribute survives serialization.
    This solution was adopted from:
    https://pandas.pydata.org/pandas-docs/stable/development/extending.html#define-original-properties
    """
    _metadata = ['seq_info_dict']

    @property
    def _constructor(self):
        return DataFrameWSeqInfo


class MOTSeqProcessor:

    def __init__(self, dataset_path, seq_name, dataset_params, cnn_model=None, logger=None, device=None):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.seq_type = _SEQ_TYPES[seq_name]

        self.det_df_loader = _SEQ_TYPE_DETS_DF_LOADER[self.seq_type]
        self.dataset_params = dataset_params

        self.cnn_model = cnn_model
        self.device = device
        self.logger = logger

    def _get_det_df(self):
        """
        Loads a pd.DataFrame where each row contains a detections bounding box' coordinates information (self.det_df),
        and, if available, a similarly structured pd.DataFrame with ground truth boxes.
        It also adds seq_info_dict as an attribute to self.det_df, containing sequence metainformation (img size,
        fps, whether it has ground truth annotations, etc.)
        """
        self.det_df, seq_info_dict, self.gt_df = self.det_df_loader(self.seq_name, self.dataset_path,
                                                                    self.dataset_params)

        self.det_df = DataFrameWSeqInfo(self.det_df)
        self.det_df.seq_info_dict = seq_info_dict

        # Some further processing
        if self.seq_type in _ENSURE_BOX_IN_FRAME and _ENSURE_BOX_IN_FRAME[self.seq_type]:
            self._ensure_boxes_in_frame()

        # Add some additional box measurements that might be used for graph construction
        self.det_df['bb_bot'] = (self.det_df['bb_top'] + self.det_df['bb_height']).values
        self.det_df['bb_right'] = (self.det_df['bb_left'] + self.det_df['bb_width']).values
        self.det_df['feet_x'] = self.det_df['bb_left'] + 0.5 * self.det_df['bb_width']
        self.det_df['feet_y'] = self.det_df['bb_top'] + self.det_df['bb_height']

        # Just a sanity check. Sometimes there are boxes that lay completely outside the frame
        frame_height, frame_width = self.det_df.seq_info_dict['frame_height'], self.det_df.seq_info_dict['frame_width']
        conds = (self.det_df['bb_width'] > 0) & (self.det_df['bb_height'] > 0)
        conds = conds & (self.det_df['bb_right'] > 0) & (self.det_df['bb_bot'] > 0)
        conds = conds & (self.det_df['bb_left'] < frame_width) & (self.det_df['bb_top'] < frame_height)
        self.det_df = self.det_df[conds].copy()

        self.det_df.sort_values(by='frame', inplace=True)
        self.det_df['detection_id'] = np.arange(self.det_df.shape[0])  # This id is used for future tastks
        # print(seq_info_dict)
        return self.det_df

    def _store_df(self):
        """
        Stores processed detections DataFrame in disk.
        """
        processed_dets_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data', 'det')
        os.makedirs(processed_dets_path, exist_ok=True)
        det_df_path = osp.join(processed_dets_path, self.det_df.seq_info_dict['det_file_name'] + '.pkl')
        self.det_df.to_pickle(det_df_path)
        print(f"Finished processing detections for seq {self.seq_name}. Result was stored at {det_df_path}")

    def _assign_gt(self):
        """
        Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
        The assignment is done frame by frame via bipartite matching.
        """
        if self.det_df.seq_info_dict['has_gt'] and not self.det_df.seq_info_dict['is_gt']:
            print(f"Assigning ground truth identities to detections to sequence {self.seq_name}")
            for frame in self.det_df['frame'].unique():
                frame_detects = self.det_df[self.det_df.frame == frame]
                frame_gt = self.gt_df[self.gt_df.frame == frame]

                # Compute IoU for each pair of detected / GT bounding box
                iou_matrix = iou(frame_detects[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values,
                                 frame_gt[['bb_top', 'bb_left', 'bb_bot', 'bb_right']].values)

                iou_matrix[iou_matrix < self.dataset_params['gt_assign_min_iou']] = np.nan
                dist_matrix = 1 - iou_matrix
                assigned_detect_ixs, assigned_detect_ixs_ped_ids = solve_dense(dist_matrix)
                unassigned_detect_ixs = np.array(list(set(range(frame_detects.shape[0])) - set(assigned_detect_ixs)))

                assigned_detect_ixs_index = frame_detects.iloc[assigned_detect_ixs].index
                assigned_detect_ixs_ped_ids = frame_gt.iloc[assigned_detect_ixs_ped_ids]['id'].values
                unassigned_detect_ixs_index = frame_detects.iloc[unassigned_detect_ixs].index

                self.det_df.loc[assigned_detect_ixs_index, 'id'] = assigned_detect_ixs_ped_ids
                self.det_df.loc[unassigned_detect_ixs_index, 'id'] = -1  # False Positives

    def _store_embeddings(self):
        """
        Stores node and reid embeddings corresponding for each detection in the given sequence.
        Embeddings are stored at:
        {seq_info_dict['seq_path']}/processed_data/embeddings/{seq_info_dict['det_file_name']}/dataset_params['node/reid_embeddings_dir'}/FRAME_NUM.pt
        Essentially, each set of processed detections (e.g. raw, prepr w. frcnn, prepr w. tracktor) has a storage path, corresponding
        to a detection file (det_file_name). Within this path, different CNNs, have different directories
        (specified in dataset_params['node_embeddings_dir'] and dataset_params['reid_embeddings_dir']), and within each
        directory, we store pytorch tensors corresponding to the embeddings in a given frame, with shape
        (N, EMBEDDING_SIZE), where N is the number of detections in the frame.
        """
        from time import time
        assert self.cnn_model is not None
        assert self.dataset_params['reid_embeddings_dir'] is not None and self.dataset_params[
            'node_embeddings_dir'] is not None

        # Create dirs to store embeddings
        node_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                    self.det_df.seq_info_dict['det_file_name'],
                                    self.dataset_params['node_embeddings_dir'])

        reid_embeds_path = osp.join(self.det_df.seq_info_dict['seq_path'], 'processed_data/embeddings',
                                    self.det_df.seq_info_dict['det_file_name'],
                                    self.dataset_params['reid_embeddings_dir'])

        if osp.exists(node_embeds_path):
            print("Found existing stored node embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(node_embeds_path)

        if osp.exists(reid_embeds_path):
            print("Found existing stored reid embeddings. Deleting them and replacing them for new ones")
            shutil.rmtree(reid_embeds_path)

        os.makedirs(node_embeds_path)
        os.makedirs(reid_embeds_path)

        # Compute and store embeddings
        # If there are more than 100k detections, we split the df into smaller dfs avoid running out of RAM, as it
        # requires storing all embedding into RAM (~6 GB for 100k detections)

        print(f"Computing embeddings for {self.det_df.shape[0]} detections")

        num_dets = self.det_df.shape[0]
        max_dets_per_df = int(1e5)  # Needs to be larger than the maximum amount of dets possible to have in one frame

        frame_cutpoints = [self.det_df.frame.iloc[i] for i in np.arange(0, num_dets, max_dets_per_df, dtype=int)]
        frame_cutpoints += [self.det_df.frame.iloc[-1] + 1]

        for frame_start, frame_end in zip(frame_cutpoints[:-1], frame_cutpoints[1:]):
            t = time()
            sub_df_mask = self.det_df.frame.between(frame_start, frame_end - 1)
            sub_df = self.det_df.loc[sub_df_mask]

            print(sub_df.frame.min(), sub_df.frame.max())
            bbox_dataset = BoundingBoxDataset(sub_df, seq_info_dict=self.det_df.seq_info_dict,
                                              return_det_ids_and_frame=True)
            bbox_loader = DataLoader(bbox_dataset, batch_size=self.dataset_params['img_batch_size'], pin_memory=True,
                                     num_workers=4)

            # Feed all bboxes to the CNN to obtain node and reid embeddings
            self.cnn_model.eval()
            node_embeds, reid_embeds = [], []
            frame_nums, det_ids = [], []
            with torch.no_grad():
                for frame_num, det_id, bboxes in bbox_loader:
                    node_out, reid_out = self.cnn_model(bboxes.to(self.device))
                    node_embeds.append(node_out.cpu())
                    reid_embeds.append(reid_out.cpu())
                    frame_nums.append(frame_num)
                    det_ids.append(det_id)
            print("IT TOOK ", time() - t)
            print(f"Finished computing embeddings")

            det_ids = torch.cat(det_ids, dim=0)
            frame_nums = torch.cat(frame_nums, dim=0)

            node_embeds = torch.cat(node_embeds, dim=0)
            reid_embeds = torch.cat(reid_embeds, dim=0)

            # Add detection ids as first column of embeddings, to ensure that embeddings are loaded correctly
            node_embeds = torch.cat((det_ids.view(-1, 1).float(), node_embeds), dim=1)
            reid_embeds = torch.cat((det_ids.view(-1, 1).float(), reid_embeds), dim=1)

            # Save embeddings grouped by frame
            for frame in sub_df.frame.unique():
                mask = frame_nums == frame
                frame_node_embeds = node_embeds[mask]
                frame_reid_embeds = reid_embeds[mask]

                frame_node_embeds_path = osp.join(node_embeds_path, f"{frame}.pt")
                frame_reid_embeds_path = osp.join(reid_embeds_path, f"{frame}.pt")

                torch.save(frame_node_embeds, frame_node_embeds_path)
                torch.save(frame_reid_embeds, frame_reid_embeds_path)

            # print("Finished storing embeddings")
        print("Finished computing and storing embeddings")

    def process_detections(self):
        # See class header
        self._get_det_df()
        # self._assign_gt()
        # self._store_df()
        # self._store_embeddings()
        return self.det_df

    def load_or_process_detections(self):
        """
        Tries to load a set of processed detections if it's safe to do so. otherwise, it processes them and stores the
        result
        """
        # Check if the processed detections file already exists.
        seq_path = osp.join(self.dataset_path, self.seq_name)
        det_file_to_use = self.dataset_params['det_file_name'] if not self.seq_name.endswith('GT') else 'gt'
        seq_det_df_path = osp.join(seq_path, 'processed_data/det', det_file_to_use + '.pkl')

        # If loading precomputed embeddings, check if embeddings have already been stored (otherwise, we need to process dets again)
        node_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use,
                                    self.dataset_params['node_embeddings_dir'])
        reid_embeds_path = osp.join(seq_path, 'processed_data/embeddings', det_file_to_use,
                                    self.dataset_params['reid_embeddings_dir'])
        try:
            num_frames = len(pd.read_pickle(seq_det_df_path)['frame'].unique())
            processed_dets_exist = True
        except:
            num_frames = -1
            processed_dets_exist = False

        embeds_ok = osp.exists(node_embeds_path) and len(os.listdir(node_embeds_path)) == num_frames
        embeds_ok = embeds_ok and osp.exists(reid_embeds_path) and len(os.listdir(reid_embeds_path)) == num_frames
        embeds_ok = embeds_ok or not self.dataset_params['precomputed_embeddings']
        # print(embeds_ok)
        if processed_dets_exist and embeds_ok and not self.dataset_params['overwrite_processed_data']:
            print(f"Loading processed dets for sequence {self.seq_name} from {seq_det_df_path}")
            seq_det_df = pd.read_pickle(seq_det_df_path).reset_index().sort_values(by=['frame', 'detection_id'])

        else:
            print(f'Detections for sequence {self.seq_name} need to be processed. Starting processing')
            seq_det_df = self.process_detections()

        seq_det_df.seq_info_dict['seq_path'] = seq_path
        frames = seq_det_df['frame'].unique()
        return seq_det_df,frames
#
# if __name__ == '__main__':
#     ####################################################################################################################################
#                         #               santity check
#     # data_frame_root = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/test/MOT17-01-DPM/processed_data/det/frcnn_prepr_det.pkl'
#     # df = pd.read_pickle(data_frame_root)
#     ####################################################################################################################################
#     data_root = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/train'
#     # data_root = '/home/kevinwm99/MOT/mot_neural_solver/data/MOT17Labels/test'
#     cnn_model = resnet50_fc256(10, loss='xent', pretrained=True)
#     torch.load(cnn_params['model_weights_path'],map_location=device)
#     load_pretrained_weights(cnn_model,cnn_params['model_weights_path'],device=device)
#     cnn_model.return_embeddings = True
#     cnn_model.to(device)
#     # model =
#
#     for seq_name in sorted(mot17_seqs):
#
#         processor = MOTSeqPRocessor(data_root,seq_name,dataset_para,cnn_model,device=device)
#         df=processor.process_detections()
#
