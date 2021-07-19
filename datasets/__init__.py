# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .mot import build as build_mot
from .mix import build as build_mix
from .crowdhuman import build as build_crowdhuman
from .roaddamage import build as build_roaddamage
from .kitti import build as build_kitti


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'mot':
        return build_mot(image_set, args)
    if args.dataset_file == 'mix':
        return build_mix(image_set, args)
    if args.dataset_file == 'crowdhuman':
        return build_crowdhuman(image_set, args)
    if args.dataset_file == 'roaddamage':
        return build_roaddamage(image_set, args)
    if (args.dataset_file == 'kitti') or (args.dataset_file == 'kitti_origin'):
        return build_kitti(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

