#!/usr/bin/env bash

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py  --output_dir ./output/roaddam20eb3_newdata --dataset_file roaddamage --coco_path ../dataset/RoadDamage --batch_size 3  --with_box_refine --num_queries 500 --epochs 20 --lr_drop 10