# SAM-AFNet

SAM-AFNet designed for paper : Automatic Measurement System for Aircraft Rivet Flushness on Surfaces Empowered by Multi-modal Large-scale Models.



Highlight:

![](https://cdn.nlark.com/yuque/0/2025/png/29220199/1754471453986-f5a8cc7f-7d65-4db4-9ae2-dbd92e1b929b.png)

# Environment
---

This code was implemented with Python 3.8.10, PyTorch 2.1.0 and CUDA version 11.8.

You can install all the requirements via:

`pip install -r requirements.txt`

# Demo

We provide an inference demo:

1. 



# Quick start

1. Download the dataset and put it in ./load (The format of the data is as follows).

```lua
load/CAMO/
│
├── Images/
│     ├── Test/															# Raw input images for testing
|     |			├── 000001.jpg
│     │			└── 000005.jpg ...
│     └── Train/														# Raw input images for training
|     			├── 000000.jpg
│     			└── 000002.jpg ...
│ 
├── PointClouds/														
│     ├── Test/															# Raw input point clouds for testing
│     │			└── labels/
│     │     			├── 000001_label.txt			# !format: [x,y,z,scalars,curvature,surface_variation]
│     │     			└── 000005_label.txt ...	# !format: [x,y,z,scalars,curvature,surface_variation]
│     └── Train/														# Raw input point clouds for training
│     			└── labels/
│           			├── 000000_label.txt
│           			└── 000002_label.txt ...
│
├── Test_gt/																# The gt masks corresponding to raw input images used for testing
│     ├── 000001.png
│     └── 000005.png ...
│
└── Train_gt/																# The gt masks corresponding to raw input images used for training
      ├── 000000.png
      └── 000002.png ...
```

2. Download the pre-trained ViT weights (sam_vit_h_4b8939.pth) and put it in ./pretrained.
3. Training:

```python
bash train.sh
```

Recommend using a graphics card with 16GB or more for coding. If you encounter memory issues, try reducing the number of `model.encoder_mode.depth` in ./configs/demo.yaml accordingly.

4. Testing:

```python
bash test.sh
```

5. Print network structure:

```python
bash show_network.sh
```

# Advanced 
How 'train.sh' and 'test.sh' work:

train.sh:

```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use-env train.py  --config configs/demo.yaml --name [your_name]
```

If you use more than one graphics card, you can:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```

test.sh

```python
python test.py --config configs/demo.yaml --model [pth_path]
```

