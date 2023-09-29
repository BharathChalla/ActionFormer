#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONUNBUFFERED=1
# Omnivore
## Training
#python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 --output reproduce
python train.py configs/error_dataset_custom_feature.yaml --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 --output reproduce

## Evaluation
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 2>&1 > logs/omnivore_recordings_3s.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type recordings --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 2>&1 > logs/omnivore_recordings_4s.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 2>&1 > logs/omnivore_person_3s.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type person --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 2>&1 > logs/omnivore_person_4s.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_3s --num_frames 90 2>&1 > logs/omnivore_environment_3s.log
python eval.py configs/error_dataset_custom_feature.yaml reproduce --backbone omnivore --division_type environment --feat_folder /data/error_dataset/features/omnivore_swinB_epic_4s --num_frames 120 2>&1 > logs/omnivore_environment_4s.log


