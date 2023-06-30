# BSH-Det3D
### [Paper](https://arxiv.org/abs/2303.02000) | [Model](https://pan.baidu.com/s/14G2PhN79B1N44LP9oDMOJA) [gib7] | [video](https://www.bilibili.com/video/BV16L41117ND/?spm_id_from=333.337.search-card.all.click&vd_source=8422e423248f61d81faa38b9b476579b)
<br/>

> [IROS 2023] BSH-Det3D: Improving 3D Object Detection with BEV Shape Heatmap

## 1. Installation
### 1.1 Requirements
All the codes are tested in the following environment:
* Linux (Ubuntu 20.04)
* Python 3.8
* PyTorch 1.10.1
* CUDA 11.3

### 1.2 Install
Our implementation is based on [[OpenPCDet v0.5.2]](https://github.com/open-mmlab/OpenPCDet), so just follow their [Installation](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).

## 2. Preparation
* During training, you should generated kitti's data including the generated complete object points as mentioned in [BtcDet](https://arxiv.org/abs/2112.02205)
, download it [[here (about 31GBs)]](https://drive.google.com/drive/folders/1mK4akt3Qro9nbw_NRfP__p2nb3a_rzxv?usp=sharing)  and put the zip file inside data/kitti/ .

* If you only want to test BSH-Det3D, please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as [GETTING_STARTED](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md).

## 3. Run Training
```
cd tools/

python train.py --cfg_file ./cfgs/kitti_models/voxelrcnn_bsh.yaml --batch_size 8
```

## 4. Run Testing
### [Model](https://pan.baidu.com/s/14G2PhN79B1N44LP9oDMOJA) [gib7]
```
cd tools/

python test.py --cfg_file ./cfgs/kitti_models/voxelrcnn_bsh.yaml --batch_size 1 --ckpt ../ckpt/bsh_voxelrcnn.pth
```

## 5. Note
If you find that low GPU utilization affects the efficiency of your model's real-time performance, try using the command:
```
export OMP_NUM_THREADS=1
```

## 6. Acknowledgement
We sincerely appreciate the following open-source projects for providing valuable and high-quality codes:
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [BtcDet](https://github.com/Xharlie/BtcDet)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)

## Citation
If you find this project useful in your research, please consider cite:
```
@article{shen2023bsh,
  title={BSH-Det3D: Improving 3D Object Detection with BEV Shape Heatmap},
  author={Shen, You and Zhang, Yunzhou and Wu, Yanmin and Wang, Zhenyu and Yang, Linghao and Coleman, Sonya and Kerr, Dermot},
  journal={arXiv preprint arXiv:2303.02000},
  year={2023}
}
```