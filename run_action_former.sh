#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
# Feature Extractors = [omnivore, videomae, slowfast, 3dresnet, x3d]
## Training
#python train.py configs/error_dataset_custom_feature.yaml --backbone videomae --division_type recordings --feat_folder /data/error_dataset/features/videomae --num_frames 30 --output reproduce
#python train.py configs/error_dataset_custom_feature.yaml --backbone videomae --division_type person --feat_folder /data/error_dataset/features/videomae --num_frames 30 --output reproduce
#python train.py configs/error_dataset_custom_feature.yaml --backbone videomae --division_type environment --feat_folder /data/error_dataset/features/videomae --num_frames 30 --output reproduce

## Evaluation
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type recordings --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type all 2>&1 > logs/videomae_recordings_all.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type recordings --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type normal 2>&1 > logs/videomae_recordings_normal.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type recordings --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type error 2>&1 > logs/videomae_recordings_error.log

python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type person --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type all 2>&1 > logs/videomae_person_all.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type person --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type normal 2>&1 > logs/videomae_person_normal.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type person --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type error 2>&1 > logs/videomae_person_error.log

python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type environment --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type all 2>&1 > logs/videomae_environment_all.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type environment --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type normal 2>&1 > logs/videomae_environment_normal.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone videomae --division_type environment --feat_folder /data/error_dataset/features/videomae --num_frames 30 --videos_type error 2>&1 > logs/videomae_environment_error.log

# Omnivore
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 2>&1 > logs/omnivore_recordings_3s.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 2>&1 > logs/omnivore_recordings_4s.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 2>&1 > logs/omnivore_person_3s.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 2>&1 > logs/omnivore_person_4s.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 2>&1 > logs/omnivore_environment_3s.log
#python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 2>&1 > logs/omnivore_environment_4s.log


