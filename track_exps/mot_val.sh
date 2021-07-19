#!/usr/bin/env bash

python3 main_track.py  --output_dir ./output/mot20b --dataset_file mot --coco_path ../dataset/MOT17 --batch_size 1 --resume ./mot20.pth --eval --with_box_refine --num_queries 300