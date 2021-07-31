

#### Requirements
- Linux, CUDA>=9.2, GCC>=5.4
- Python>=3.7
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization


#### Steps
1. Install and build libs
```
git clone https://github.com/sYANGunique1111/transtrack_kitti.git
cd transtrack_kitti
cd models/ops
python setup.py build install
cd ../..
pip install -r requirements.txt
```

2. Prepare datasets and annotations
```
mkdir ../dataset/kitti
cp -r /path_to_kitti_dataset/train kitti/train
cp -r /path_to_kitti_dataset/test kitti/test
```
Kitti dataset is available in [Kitti](http://www.cvlibs.net/datasets/kitti/). 
```
python3 track_tools/convert_kitti_to_coco.py
```
The pre-trained model is available [crowdhuman_final.pth](https://drive.google.com/drive/folders/1DjPL8xWoXDASrxgsA3O06EspJRdUXFQ-?usp=sharing).

4. Train TransTrack
```
sh track_exps/kitti_trainhalf.sh
```

5. Evaluate TransTrack
```
sh track_exps/kitti_val.sh
sh track_exps/mota.sh
```

6. Visualize TransTrack
```
python3 track_tools/txt2video.py
```



## License

TransTrack is released under MIT License.

