#!/usr/bin/env bash


GROUNDTRUTH= '/home/syang/stage/dataset/MOT17/train/'
RESULTS= '/home/syang/stage/transtrack_origin/evaluation/mot20b/val/tracks'
GT_TYPE=_val_half
THRESHOLD=-1

python3 track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH} \
--tests ${RESULTS} \
--gt_type ${GT_TYPE} \
--eval_official \
--score_threshold ${THRESHOLD}
