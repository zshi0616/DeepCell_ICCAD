from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
import deepcell.top_model
import deepcell.top_trainer 
from torch_geometric.data import Data
import deepgate as dg

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_sample'

if __name__ == '__main__':
    args = get_parse_args()
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, random_sample=1.0)
    train_dataset, val_dataset = dataset.get_dataset()
    
    print('[INFO] Create Model and Trainer')
    model = deepcell.top_model.TopModel(
        args, 
        pm_ckpt='./ckpt/pm_{}.pth'.format(args.pm_aggr), 
        aig_ckpt='./ckpt/aig_{}.pth'.format(args.aig_encoder)
    )

    trainer = deepcell.top_trainer.TopTrainer(args, model, distributed=args.distributed, device=args.device)
    if args.resume:
        trainer.resume()
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1, 5])
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    
