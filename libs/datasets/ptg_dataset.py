import os
import json
import numpy as np
import pandas as pd

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats


@register_dataset("ptg_dataset")
class PTGDataset(Dataset):
    def __init__(
            self,
            is_training,  # if in training mode
            split,  # split, a tuple/list allowing concat of subsets
            feat_folder,  # folder for features
            json_file,  # json file for annotations
            feat_stride,  # temporal stride of the feats
            num_frames,  # number of frames for each feat
            default_fps,  # default fps
            downsample_rate,  # downsample rate for feats
            max_seq_len,  # maximum sequence length during training
            trunc_thresh,  # threshold for truncate an action segment
            crop_ratio,  # a tuple (e.g., (0.9, 1.0)) for random cropping
            input_dim,  # input feat dim
            num_classes,  # number of action categories
            file_prefix,  # feature file prefix if any
            file_suffix,  # feature file prefix if any
            file_ext,  # feature file extension if any
            force_upsampling,  # force to upsample to max_seq_len
            backbone,  # backbone type
            division_type,  # data division type
            videos_type,  # videos type (all, error, normal)
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.backbone = backbone
        self.division_type = division_type
        self.videos_type = videos_type

        self.file_prefix = file_prefix if file_prefix is not None else ''
        self.file_suffix = file_suffix if file_suffix is not None else '_360p'
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file, videos_type)
        # "empty" noun categories on epic-kitchens
        assert len(label_dict) <= num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'ptg-dataset',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': empty_label_ids
        }

    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids

    def get_attributes(self):
        return self.db_attributes

    @staticmethod
    def get_file_key(key):
        rid = key.split("_")[0]
        old_recipes_dict = {
            '5': 'RecipeB',
            '10': 'RecipeA',
        }
        file_key = key
        if rid in old_recipes_dict:
            rec_id = key.split("_")[1]
            rec_id = int(rec_id) - 1
            rid = old_recipes_dict[rid]
            file_key = f"{rid}_{rec_id}"
        return file_key

    def is_skip_video(self, key, videos_type):
        # skip normal videos
        # skip_normal, skip_error = False, False  # All videos (Error + Normal)
        # skip_normal, skip_error = True, False   # All Error videos
        # skip_normal, skip_error = False, True  # All Normal videos
        # skip_normal, skip_error = True, True    # No videos
        if videos_type == "all":
            skip_normal, skip_error = False, False  # All videos (Error + Normal)
        elif videos_type == "error":
            skip_normal, skip_error = True, False
        elif videos_type == "normal":
            skip_normal, skip_error = False, True
        else:
            skip_normal, skip_error = True, True
        recipe_has_error_id = int(key.split("_")[1])
        if skip_normal and (1 <= recipe_has_error_id <= 25 or 100 <= recipe_has_error_id <= 125):
            return True
        if skip_error and (26 <= recipe_has_error_id <= 99 or 126 <= recipe_has_error_id <= 200):
            return True
        return False

    def _load_json_db(self, json_file, videos_type='all'):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable after wards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            features_dir = os.path.join(self.feat_folder, self.file_prefix + key + self.file_suffix)
            if self.backbone != 'omnivore' or self.num_frames == 30:
                self.file_ext = '.npy'
            feat_file = os.path.join(features_dir, f"video_features{self.file_ext}")
            if not os.path.exists(feat_file):
                continue

            if self.is_skip_video(key, videos_type):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps': fps,
                         'duration': duration,
                         'segments': segments,
                         'labels': labels
                         },)

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preprocess the data
        video_item = self.data_list[idx]

        # load features
        filename = os.path.join(self.feat_folder, video_item['id'] + self.file_suffix, "video_features" + self.file_ext)
        if self.file_ext == '.npz':
            with np.load(filename) as data:
                feats = data['video_features'].astype(np.float32)
        elif self.file_ext == '.npy':
            feats = np.load(filename).astype(np.float32)

        # Todo: Continue from here
        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id': video_item['id'],
                     'feats': feats,  # C x T
                     'segments': segments,  # N x 2
                     'labels': labels,  # N
                     'fps': video_item['fps'],
                     'duration': video_item['duration'],
                     'feat_stride': feat_stride,
                     'feat_num_frames': self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )

        return data_dict
