#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 10:35:11 2021

@author: syang
"""

import os
import numpy as np
import json
from PIL import Image
import glob
import xml.etree.ElementTree as ET


DATA_PATH = '../../dataset/RoadDamage'
SUB_FOLDERS = ['Japan', 'India', 'Czech']
OUT_PATH = os.path.join(DATA_PATH, 'annotations_coco/')
SPLITS = ['train_half_80', 'val_half_20', 'train']  # --> split training data to train_half and val_half.
NAMES = ['D00', 'D01', 'D10', 'D11', 'D20',
         'D40', 'D43', 'D44', 'D50']
FILES = []
OUT = []
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CREATE_SPLITTED_DET = True


def getFilesInDir(dir_path):
    assert os.path.exists(dir_path), f'{dir_path}. The annotation path doesn\'t exist'
    file_list = sorted(glob.glob(dir_path + '/*.xml'))
    return file_list

def read_content(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()
    if (root.find('object') is None) or (root.find('object/name').text == 'D0w0'):    
        return 0
    
    return root

def load_func(fpath):
    print('fpath', fpath)
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records =[json.loads(line.strip('\n')) for line in lines]
    return records

if __name__ == '__main__':
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:           
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        data_path = DATA_PATH + split
        out_path = OUT_PATH + '{}.json'.format(split)
        #out = {'images': [], 'annotations': [], 'categories': [{'id': [0,1,2,3,4,5,6,7,8], 'name': 'roaddamage'}]}
        out = {'images': [], 'annotations': [],
                 'categories': [{'id': 0, 'name': 'D00'},
                                {'id': 1, 'name': 'D01'},
                                {'id': 2, 'name': 'D10'},
                                {'id': 3, 'name': 'D11'},
                                {'id': 4, 'name': 'D20'},
                                {'id': 5, 'name': 'D40'},
                                {'id': 6, 'name': 'D43'},
                                {'id': 7, 'name': 'D44'},
                                {'id': 8, 'name': 'D50'}]}

        for sub_dir in SUB_FOLDERS:
            ann_path = os.path.join(DATA_PATH, 'train', sub_dir, 'annotations/xmls')
            num_images = len(getFilesInDir(ann_path))
            if 'half' in split:
                image_range = [0, num_images * 4 // 5] if 'train' in split else \
                                 [num_images * 4 // 5 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]
                
            for i, ann_ in enumerate(getFilesInDir(ann_path)):
                if i < image_range[0] or i > image_range[1]:
                    continue
                image_cnt += 1
                root = read_content(ann_)
                if root == 0:
                    continue
                file_name = root.find('filename').text
                height = int(root.find('size/height').text)
                width = int(root.find('size/width').text)
                image_info = {'file_name': '{}/JPEGImages/{}'.format(sub_dir, file_name), 
                                    'id': image_cnt,
                                    'height': height, 
                                    'width': width}
                out['images'].append(image_info)
                for obj in root.iter('object'):
                    ann_cnt += 1
                    if 'difficult' in obj.keys():
                        difficult = obj.find('difficult').text
                    else:
                        difficult = 0
                    obj_name = obj.find('name').text
                    cls_id = NAMES.index(obj_name)
                    xmlbox = obj.find('bndbox')
                    b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]
                    bb = [b[0], b[1], b[2]-b[0], b[3]-b[1]]
                    ann = {'id': ann_cnt,
                         'category_id': cls_id,
                         'image_id': image_cnt,
                         'bbox': bb,
                         'area': bb[2] * bb[3],
                         'iscrowd': 0}
                    out['annotations'].append(ann)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))