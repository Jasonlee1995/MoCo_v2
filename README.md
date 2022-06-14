# MoCo v2 Implementation with Pytorch
- Unofficial implementation of the paper *Improved Baselines with Momentum Contrastive Learning*


## 0. Develop Environment
- Check `env.txt` for more details


## 1. Implementation Details
- data.py : data augmentations, dataset
- main_linear.py : only train linear classifier with frozen backbone
- main_pretrain.py : pre-train backbone
- models_linear.py : model for linear probing
- models_moco.py : model for pre-train
- utils.py : utils such as scheduler, logger
- logs : log files
- visualize : visualize pretrain, linear probing log (acc, lr, train loss, val loss)


## 2. Linear Probing Result Comparison on ImageNet
|Source|Score|Detail|
|:-:|:-:|:-|
|Paper|67.7|200 epochs|
|Paper|71.1|800 epochs|
|Current Repo|67.41|200 epochs, query|
|Current Repo|67.50|200 epochs, key|
|Current Repo|70.12|800 epochs, query|
|Current Repo|70.15|800 epochs, key|


## 3. Reference
- Improved Baselines with Momentum Contrastive Learning [[paper](https://arxiv.org/abs/2003.04297)] [[official code](https://github.com/facebookresearch/moco)]
