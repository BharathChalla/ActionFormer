import os

import numpy as np


def combine_features():
    backbone = 'videomae'
    features_root_dir = "/data/error_dataset/features"

    one_sec_features_dir = os.path.join(features_root_dir, "video_mae")
    features_save_dir = os.path.join(features_root_dir, backbone)
    for rec_id in sorted(os.listdir(one_sec_features_dir)):
        rec_dir = os.path.join(one_sec_features_dir, rec_id)
        # natural sorting
        time_sorted_rec_dirs = sorted(os.listdir(rec_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))
        video_feat = []
        print(f"Processing {rec_id}...")
        for feat_file in time_sorted_rec_dirs:
            feat_file_path = os.path.join(rec_dir, feat_file)
            feat = np.load(feat_file_path)
            video_feat.append(feat)
        video_feat = np.stack(video_feat, axis=0)
        video_feat_path = os.path.join(features_save_dir, rec_id, 'video_features.npy')
        os.makedirs(os.path.dirname(video_feat_path), exist_ok=True)
        np.save(video_feat_path, video_feat)
        print(f"Saved {rec_id}: {video_feat_path}")
    pass


if __name__ == '__main__':
    combine_features()
