
![image info](./property/results.png)
Thats result.
## Environment
1. cuda>=12 (not necessary)
2. GPU memory>=12G
3. pytorch>=2.0.1


## Quick Start
1. conda create -n your_name python==3.12
2. conda activate your_name
3. conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
4. pip install -r requirement.txt

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
<!-- - Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- The training command is like -->
For multi GPUs, the training comand is like:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1231 Wnet/train.py -opt ./options/train/turbulence_deblur.yml --launcher pytorch
```

For single GPU, the training command is like:
```
python Wnet/train.py -opt ./options/train/turbulence_deblur.yml
```

The training logs and weights will be saved in the `./experiments` folder.

## Resource consuption
1. Deblur and Detilt task
- Note that the default batch size per GPU is 4, which will cost about 16G memory for each GPU.
2. Deturb tasl
- Note that the default batch size per GPU is 1, which will cost about 16G memory for each GPU.