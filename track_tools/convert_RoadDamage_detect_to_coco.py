#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:00:16 2021

@author: syang
"""

import glob
import os
import numpy as np
import json
import cv2


# Use the same script for MOT16
DATA_PATH = '../../dataset/RoadDamage/RoadDamageDataset'
OUT_PATH = os.path.join(DATA_PATH, 'annotations_coco')
SPLITS = ['train_half_80', 'val_half_20', 'train']  # --> split training data to train_half and val_half.
NAMES = ['D00', 'D01', 'D10', 'D11', 'D20',
         'D40', 'D43', 'D44']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test')
        else:
            data_path = os.path.join(DATA_PATH, 'train')
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': [0,1,2,3,4,5,6,7], 'name': NAMES}]}
        seqs = os.listdir(data_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        for seq in sorted(seqs):
            seq_path = os.path.join(data_path, seq)
            img_path = os.path.join(seq_path, 'JPEGImages')
            ann_path = os.path.join(seq_path, 'gt_labels/{}_gt.txt'.format(seq))
            if not (os.path.exists(ann_path) and os.path.exists(ann_path)):
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            anns_frames = np.loadtxt(ann_path, dtype=np.int32, delimiter=',', usecols=(0))
            images = os.listdir(img_path)
            images = sorted(glob.glob(img_path + '/*.jpg'))
            images = [image.split('/')[-1] for image in images]
            num_images = len(images) 
            #num_images = len([image for image in images if 'jpg' in image])  # half and half

            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images * 4 // 5] if 'train' in split else \
                              [num_images * 4 // 5 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]

            #for i in range(num_images):
            for i, image in enumerate(images):
                if i < image_range[0] or i > image_range[1]:
                    continue
                if i not in anns_frames :
                    continue
                img = cv2.imread(os.path.join(data_path, '{}/JPEGImages/{}'.format(seq, image)))
                height, width = img.shape[:2]
                image_info = {'file_name': '{}/JPEGImages/{}'.format(seq, image),  # image name.
                                'id': image_cnt + i + 1,
                                'height': height, 
                                'width': width}
                  

                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))
            if split != 'test':
                #det_path = os.path.join(seq_path, 'det/det.txt')
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')
                
                #dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')
                if CREATE_SPLITTED_ANN and ('half' in split):
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32) 
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, 'gt_labels/gt_{}.txt'.format(split))
                    fout = open(gt_out, 'w')
                    for o in anns_out:
                        fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                    int(o[6]), int(o[7]), o[8]))
                    fout.close()
                '''
                if CREATE_SPLITTED_DET and ('half' in split):
                    dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                         if int(dets[i][0])  >= image_range[0] and
                                         int(dets[i][0])  <= image_range[1]], np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, 'det/det_{}.txt'.format(split))
                    dout = open(det_out, 'w')
                    for o in dets_out:
                        dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]),
                                    float(o[6])))
                    dout.close()
                '''
                print('{} ann images'.format(int(anns[:, 0].max())))
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0]) + 1
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])
                    category_id = int(anns[i][6])
                    ann_cnt += 1
                    '''
                    if not ('15' in DATA_PATH):
                        if not (float(anns[i][8]) >= 0.25):  # visibility.
                            continue
                        if not (int(anns[i][6]) == 1):  # whether ignore.
                            continue
                        if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                            continue
                        if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
                            category_id = -1
                        else:
                            category_id = 1  # pedestrian(non-static)
                    else:
                        category_id = 1
                    '''
                    
                    ann = {'id': ann_cnt,
                         'category_id': category_id,
                         'image_id': image_cnt + frame_id ,
                         'bbox': anns[i][2:6].tolist(),
                         'area': float(anns[i][4] * anns[i][5]),
                         'category_name': NAMES[category_id],
                         'iscrowd': 0 }
                    
                    '''
                    ann = {'id': ann_cnt,
                           #'category_id': category_id,
                           'category_id': category_id,
                           'image_id': image_cnt + frame_id,
                           #'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           #'conf': float(anns[i][6]),
                           'conf': 1.,
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                    '''
                    out['annotations'].append(ann)
            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
        
