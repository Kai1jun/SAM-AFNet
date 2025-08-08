# SAM-AFNet

SAM-AFNet designed for paper : Automatic Measurement System for Aircraft Rivet Flushness on Surfaces Empowered by Multi-modal Large-scale Models.



Highlight:

![](https://cdn.nlark.com/yuque/0/2025/png/29220199/1754471453986-f5a8cc7f-7d65-4db4-9ae2-dbd92e1b929b.png)

# Environment

This code was implemented with Python 3.8.10, PyTorch 2.1.0 and CUDA version 11.8.

You can install all the requirements via:

```python
pip install -r requirements.txt
```

# Demo

We provide an inference demo:

1. Download the [demo data](https://drive.google.com/drive/folders/1seyVl8NeCPyNTHFSFW5_QdVtDOEiP2O5?usp=drive_link) and input it in ./load/
2. Download the [pretrained file](https://drive.google.com/file/d/1Iz_2HB79L2zZNa2APjwj9Gb0yxATpUiM/view?usp=drive_link) and input it in ./save/
3. Run test code:

```python
python test.py --config configs/demo.yaml --model save/model_epoch_last.pth
```
4. Results (50 epochs): 

 ![](https://cdn.nlark.com/yuque/0/2025/png/29220199/1754641332846-9b8a23b3-8b34-43f9-8031-9b2a0ddc6c12.png)

# Quick start

1. Download the dataset and put it in ./load (The format of the data is as follows).

```lua
load/CAMO/
│
├── Images/
│     ├── Test/				# Raw input images for testing
|     |	├── 000001.jpg
│     │	└── 000005.jpg ...
│     └── Train/			# Raw input images for training
|     	├── 000000.jpg
│     	└── 000002.jpg ...
│ 
├── PointClouds/														
│     ├── Test/				# Raw input point clouds for testing
│     │	└── labels/
│     │     	├── 000001_label.txt                  # !format: [x,y,z,scalars,curvature,surface_variation]
│     │     	└── 000005_label.txt ...              # !format: [x,y,z,scalars,curvature,surface_variation]
│     └── Train/			# Raw input point clouds for training
│     	└── labels/
│           	├── 000000_label.txt
│           	└── 000002_label.txt ...
│
├── Test_gt/				# The gt masks corresponding to raw input images used for testing
│     ├── 000001.png
│     └── 000005.png ...
│
└── Train_gt/				# The gt masks corresponding to raw input images used for training
      ├── 000000.png
      └── 000002.png ...
```

2. Download the pre-trained ViT weights (sam_vit_h_4b8939.pth) and put it in ./pretrained.
3. Training:

```python
bash train.sh
```

Recommend using a graphics card with 16GB or more for coding. If you encounter memory issues, try reducing the number of `model.encoder_mode.depth` in `./configs/demo.yaml` accordingly.

4. Testing:

```python
bash test.sh
```

5. Print network structure:

```python
bash show_network.sh
```
The structure of the network will be printed on the terminal: 
```python
=========================================================================================================
Layer (type:depth-idx)                                  Output Shape              Param #
=========================================================================================================
SAMWrapper                                              --                        --
├─SAM: 1-1                                              --                        256
│    └─Prompt3dEncoder: 2-1                             [1, 0, 256]               256
│    │    └─Sequential: 3-1                             [1, 256, 64, 64]          --
│    │    │    └─Conv2d: 4-1                            [1, 8, 64, 64]            80
│    │    │    └─LayerNorm2d: 4-2                       [1, 8, 64, 64]            16
│    │    │    └─GELU: 4-3                              [1, 8, 64, 64]            --
│    │    │    └─Conv2d: 4-4                            [1, 16, 64, 64]           1,168
│    │    │    └─LayerNorm2d: 4-5                       [1, 16, 64, 64]           32
│    │    │    └─GELU: 4-6                              [1, 16, 64, 64]           --
│    │    │    └─Conv2d: 4-7                            [1, 32, 64, 64]           4,640
│    │    │    └─LayerNorm2d: 4-8                       [1, 32, 64, 64]           64
│    │    │    └─GELU: 4-9                              [1, 32, 64, 64]           --
│    │    │    └─Conv2d: 4-10                           [1, 64, 64, 64]           18,496
│    │    │    └─LayerNorm2d: 4-11                      [1, 64, 64, 64]           128
│    │    │    └─GELU: 4-12                             [1, 64, 64, 64]           --
│    │    │    └─Conv2d: 4-13                           [1, 128, 64, 64]          73,856
│    │    │    └─LayerNorm2d: 4-14                      [1, 128, 64, 64]          256
│    │    │    └─GELU: 4-15                             [1, 128, 64, 64]          --
│    │    │    └─Conv2d: 4-16                           [1, 256, 64, 64]          33,024
│    └─ImageEncoderViT: 2-2                             [1, 256, 64, 64]          5,242,880
│    │    └─PatchEmbed: 3-2                             [1, 64, 64, 1280]         --
│    │    │    └─Conv2d: 4-17                           [1, 1280, 64, 64]         984,320
│    │    └─PromptGenerator: 3-3                        --                        --
│    │    │    └─Linear: 4-18                           [1, 4096, 40]             51,240
│    │    │    └─PatchEmbed2: 4-19                      [1, 40, 64, 64]           --
│    │    │    │    └─Conv2d: 5-1                       [1, 40, 64, 64]           30,760
│    │    │    └─Sequential: 4-20                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-2                       [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-3                         [1, 4096, 40]             --
│    │    │    └─Linear: 4-21                           [1, 4096, 1280]           52,480
│    │    │    └─Sequential: 4-22                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-4                       [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-5                         [1, 4096, 40]             --
│    │    │    └─Linear: 4-23                           [1, 4096, 1280]           (recursive)
│    │    │    └─Sequential: 4-24                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-6                       [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-7                         [1, 4096, 40]             --
│    │    │    └─Linear: 4-25                           [1, 4096, 1280]           (recursive)
│    │    │    └─Sequential: 4-26                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-8                       [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-9                         [1, 4096, 40]             --
│    │    │    └─Linear: 4-27                           [1, 4096, 1280]           (recursive)
│    │    │    └─Sequential: 4-28                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-10                      [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-11                        [1, 4096, 40]             --
│    │    │    └─Linear: 4-29                           [1, 4096, 1280]           (recursive)
│    │    │    └─Sequential: 4-30                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-12                      [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-13                        [1, 4096, 40]             --
│    │    │    └─Linear: 4-31                           [1, 4096, 1280]           (recursive)
│    │    │    └─Sequential: 4-32                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-14                      [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-15                        [1, 4096, 40]             --
│    │    │    └─Linear: 4-33                           [1, 4096, 1280]           (recursive)
│    │    │    └─Sequential: 4-34                       [1, 4096, 40]             --
│    │    │    │    └─Linear: 5-16                      [1, 4096, 40]             1,640
│    │    │    │    └─GELU: 5-17                        [1, 4096, 40]             --
│    │    │    └─Linear: 4-35                           [1, 4096, 1280]           (recursive)
│    │    └─ModuleList: 3-4                             --                        --
│    │    │    └─Block: 4-36                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-18                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-19                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-20                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-21                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-37                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-22                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-23                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-24                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-25                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-38                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-26                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-27                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-28                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-29                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-39                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-30                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-31                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-32                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-33                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-40                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-34                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-35                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-36                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-37                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-41                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-38                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-39                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-40                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-41                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-42                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-42                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-43                   [25, 14, 14, 1280]        6,563,040
│    │    │    │    └─LayerNorm: 5-44                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-45                    [1, 64, 64, 1280]         13,113,600
│    │    │    └─Block: 4-43                            [1, 64, 64, 1280]         --
│    │    │    │    └─LayerNorm: 5-46                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─Attention: 5-47                   [1, 64, 64, 1280]         6,579,040
│    │    │    │    └─LayerNorm: 5-48                   [1, 64, 64, 1280]         2,560
│    │    │    │    └─MLPBlock: 5-49                    [1, 64, 64, 1280]         13,113,600
│    │    └─Sequential: 3-5                             [1, 256, 64, 64]          --
│    │    │    └─Conv2d: 4-44                           [1, 256, 64, 64]          327,680
│    │    │    └─LayerNorm2d: 4-45                      [1, 256, 64, 64]          512
│    │    │    └─Conv2d: 4-46                           [1, 256, 64, 64]          589,824
│    │    │    └─LayerNorm2d: 4-47                      [1, 256, 64, 64]          512
│    └─PositionEmbeddingRandom: 2-3                     [256, 64, 64]             --
│    └─MaskDecoder: 2-4                                 [1, 1, 256, 256]          1,280
│    │    └─TwoWayTransformer: 3-6                      [1, 5, 256]               --
│    │    │    └─ModuleList: 4-48                       --                        --
│    │    │    │    └─TwoWayAttentionBlock: 5-50        [1, 5, 256]               1,579,520
│    │    │    │    └─TwoWayAttentionBlock: 5-51        [1, 5, 256]               1,579,520
│    │    │    └─Attention: 4-49                        [1, 5, 256]               --
│    │    │    │    └─Linear: 5-52                      [1, 5, 128]               32,896
│    │    │    │    └─Linear: 5-53                      [1, 4096, 128]            32,896
│    │    │    │    └─Linear: 5-54                      [1, 4096, 128]            32,896
│    │    │    │    └─Linear: 5-55                      [1, 5, 256]               33,024
│    │    │    └─LayerNorm: 4-50                        [1, 5, 256]               512
│    │    └─Sequential: 3-7                             [1, 32, 256, 256]         --
│    │    │    └─ConvTranspose2d: 4-51                  [1, 64, 128, 128]         65,600
│    │    │    └─LayerNorm2d: 4-52                      [1, 64, 128, 128]         128
│    │    │    └─GELU: 4-53                             [1, 64, 128, 128]         --
│    │    │    └─ConvTranspose2d: 4-54                  [1, 32, 256, 256]         8,224
│    │    │    └─GELU: 4-55                             [1, 32, 256, 256]         --
│    │    └─ModuleList: 3-8                             --                        --
│    │    │    └─MLP: 4-56                              [1, 32]                   --
│    │    │    │    └─ModuleList: 5-56                  --                        139,808
│    │    │    └─MLP: 4-57                              [1, 32]                   --
│    │    │    │    └─ModuleList: 5-57                  --                        139,808
│    │    │    └─MLP: 4-58                              [1, 32]                   --
│    │    │    │    └─ModuleList: 5-58                  --                        139,808
│    │    │    └─MLP: 4-59                              [1, 32]                   --
│    │    │    │    └─ModuleList: 5-59                  --                        139,808
│    │    └─MLP: 3-9                                    [1, 4]                    --
│    │    │    └─ModuleList: 4-60                       --                        --
│    │    │    │    └─Linear: 5-60                      [1, 256]                  65,792
│    │    │    │    └─Linear: 5-61                      [1, 256]                  65,792
│    │    │    │    └─Linear: 5-62                      [1, 4]                    1,028
=========================================================================================================
```

# Advanced 
How 'train.sh' and 'test.sh' work:

train.sh:

```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --use-env train.py  --config configs/demo.yaml --name [YOUR_NAME]
```

If you use more than one graphics card, you can:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```

test.sh

```python
python test.py --config configs/demo.yaml --model [PTH_PATH]
```

# Acknowledgements
Some of the code of this repo is borrowed from:
[SAMAdapterPyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch).

The part of the code is derived from [Explicit Visual Prompt](https://nifangbaage.github.io/Explicit-Visual-Prompt/) by Weihuang Liu, Xi Shen, Chi-Man Pun, and Xiaodong Cun by University of Macau and Tencent AI Lab.
