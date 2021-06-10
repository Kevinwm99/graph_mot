import torch
import matplotlib.pyplot as plt
import pandas as pd
from create_dataloader import MOTGraph
from pack import MOTSeqProcessor

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


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"
    device = torch.device('cuda:5')
    for seq_name in mot17_train:

        processor = MOTSeqProcessor(DATA_ROOT, seq_name, dataset_para, device=device)
        df, frames = processor.load_or_process_detections()
        df_len = len(df)
        max_frame_per_graph = 15
        fps = df.seq_info_dict['fps']
        # for i in range(1, len(frames)-max_frame_per_graph+2):
        #     print("Construct graph {} from frame {} to frame {}".format(seq_name,i, i+14))
        mot_graph_past = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                  start_frame=190,
                                  end_frame=190+13)
        mot_graph_future = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                    start_frame=190+14,
                                    end_frame=190+14)
        mot_graph_gt = MOTGraph(seq_det_df=df, seq_info_dict=df.seq_info_dict, dataset_params=dataset_para,
                                start_frame=190,
                                end_frame=190+14)
        print(mot_graph_past.seq_info_dict)

        break

