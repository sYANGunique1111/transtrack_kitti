#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main_track.py  --output_dir ./output/RoadDamageDataset10e2b --dataset_file roaddamage --coco_path ../dataset/RoadDamage/RoadDamageDataset --batch_size 2  --with_box_refine  --num_queries 300 --epochs 10 --lr_drop 10
