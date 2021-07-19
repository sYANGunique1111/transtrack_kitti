python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main_track.py  --output_dir ./output/kitti_new2_ori20e3b --dataset_file kitti_origin --coco_path ../dataset/Kitti_left --batch_size 3  --with_box_refine  --num_queries 300 --epochs 20  --lr 1e-4 --lr_drop 15


#python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main_track.py  --output_dir ./output/kitti_new2_ori20e3b --dataset_file kitti_origin --coco_path ../dataset/Kitti_left --batch_size 3  --with_box_refine  --num_queries 300 --epochs 20  --lr 1e-4 --lr_drop 15