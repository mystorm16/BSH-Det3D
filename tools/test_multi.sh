#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${NGPUS}

python3 test.py --cfg_file /home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}.yaml --batch_size 1 --workers 4 --ckpt /home/shenyou/btc_center/output/home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}/default/ckpt/checkpoint_epoch_70.pth &
python3 test.py --cfg_file /home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}.yaml --batch_size 1 --workers 4 --ckpt /home/shenyou/btc_center/output/home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}/default/ckpt/checkpoint_epoch_72.pth &
python3 test.py --cfg_file /home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}.yaml --batch_size 1 --workers 4 --ckpt /home/shenyou/btc_center/output/home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}/default/ckpt/checkpoint_epoch_74.pth &
python3 test.py --cfg_file /home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}.yaml --batch_size 1 --workers 4 --ckpt /home/shenyou/btc_center/output/home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}/default/ckpt/checkpoint_epoch_76.pth &
python3 test.py --cfg_file /home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}.yaml --batch_size 1 --workers 4 --ckpt /home/shenyou/btc_center/output/home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}/default/ckpt/checkpoint_epoch_78.pth &
python3 test.py --cfg_file /home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}.yaml --batch_size 1 --workers 4 --ckpt /home/shenyou/btc_center/output/home/shenyou/btc_center/tools/cfgs/kitti_models/${PY_ARGS}/default/ckpt/checkpoint_epoch_80.pth