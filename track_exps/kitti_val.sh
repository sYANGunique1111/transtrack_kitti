#!/usr/bin/env bash

python3 main_track.py  --output_dir ./evaluation/kitti_new2_20e3bval --dataset_file kitti --coco_path ../dataset/Kitti_left --batch_size 1 --resume output/kitti_new2_20e3b/checkpoint.pth --eval --with_box_refine --num_queries 300