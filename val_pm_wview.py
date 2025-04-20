from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
from torch import nn
import time
import deepcell.top_model
import deepcell.top_trainer 

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_test'

if __name__ == '__main__':
    args = get_parse_args()
    args.refine = 'pm'
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, trainval_split=0, random_shuffle=False)
    _, val_dataset = dataset.get_dataset()
    
    print('[INFO] Create Model')
    model = deepcell.top_model.TopModel(
        args, 
    )
    trainer = deepcell.top_trainer.TopTrainer(args, model, distributed=False, device='cpu')
    trainer.resume()
    trainer.train(1, train_dataset=None, val_dataset=val_dataset)
    
    