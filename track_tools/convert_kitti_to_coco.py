#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:42:02 2021

@author: syang
"""
import os
import numpy as np
import json
import cv2


# Use the same script for MOT16
DATA_PATH = '../../dataset/Kitti_left'
GT_FOLDER = 'reformed_labels/label_02_new'
OUT_PATH = os.path.join(DATA_PATH, 'annotations_origin')
CLASSES = ['Car', 'Pedestrian', 'Van', 'Misc', 'Cyclist', 'Truck', 'Person', 'Tram', 'DontCare']
#SPLITS = ['train_half', 'val_half', 'train']  # --> split training data to train_half and val_half.
SPLITS = ['val_half', 'train_half', 'train']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        data_path = os.path.join(DATA_PATH, GT_FOLDER)
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
                 'categories': [{'id': 0, 'name': 'Car'},
                                {'id': 1, 'name': 'Pedestrian'},
                                {'id': 2, 'name': 'Van'},
                                {'id': 3, 'name': 'Misc'},
                                {'id': 4, 'name': 'Cyclist'},
                                {'id': 5, 'name': 'Truck'},
                                {'id': 6, 'name': 'Person'},
                                {'id': 7, 'name': 'Tram'}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            if '.txt' not in seq:
                continue
            seq_noext = os.path.splitext(seq)[0]
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq_noext})
            ann_path = os.path.join(data_path, seq)
            img_path = os.path.join(DATA_PATH, 'train/image_02/{}'.format(seq_noext))
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'png' in image])  # half and half

            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            for i in range(num_images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                img = cv2.imread(os.path.join(img_path, '{:06d}.png'.format(i)))
                height, width = img.shape[:2]
                image_info = {'file_name': 'image_02/{}/{:06d}.png'.format(seq_noext, i),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq_noext, num_images))
            if split != 'test':
                #det_path = os.path.join(seq_path, 'det/det.txt')
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',', comments='#')
                #dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(img_path, 'gt_coco')
                    if not os.path.exists(gt_out):
                        os.mkdir(gt_out)
                    fout = open(os.path.join(gt_out, 'gt_{}.txt'.format(split)), 'w')
                    #fout.write('# frame, tracking_id, bbox(xmin,ymin,weight,height), class, -1, -1, -1\n')
                    obj_number = np.unique(anns_out[:,1])
                    #for o in anns_out[anns_out[:,1].argsort()]:
                    for num in obj_number:
                        if int(num) == -1:
                            continue
                        sub_anns = anns_out[np.where(anns_out[:,1] == num)]
                        for o in sub_anns:
                            fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:f}\n'.format(
                                        int(o[0]), int(o[1])+1, int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                        1, int(o[6]), 1))
                    fout.close()

                print('{} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    cat_id = int(anns[i][7])
                    ann_cnt += 1
                    category_id = int(anns[i][6])
                    if category_id == -1:
                        continue
                    iscrowd = 0
                    #iscrowd = 0

                    
                    ann = {'id': ann_cnt,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(1),
                           'iscrowd': iscrowd,
                           'area': float(anns[i][4] * anns[i][5])}
                    out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
        
