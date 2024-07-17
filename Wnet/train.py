# flake8: noqa
import os.path as osp
import sys
import os
sys.path.append(os.getcwd())

import Wnet.archs
import Wnet.data
import Wnet.losses
import Wnet.models
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
