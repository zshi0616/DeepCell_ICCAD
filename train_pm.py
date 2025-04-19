from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_sample'
# checkpoint = './ckpt/pm_dg2.pth'

if __name__ == '__main__':
    args = get_parse_args()
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR)
    train_dataset, val_dataset = dataset.get_dataset()
    
    print('[INFO] Create Model and Trainer')
    model = deepcell.Model(aggr=args.pm_aggr)
    # model.load(checkpoint)
    
    trainer = deepcell.Trainer(args, model, distributed=args.distributed, device=args.device, training_id=args.exp_id)
    if args.resume:
        trainer.resume()
    trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=80)
    print('[INFO] Stage 1 Training ...')
    trainer.train(40, train_dataset, val_dataset)
    
    # trainer.set_training_args(loss_weight=[3.0, 1.0, 0.5], lr=1e-4, lr_step=80)
    # print('[INFO] Stage 2 Training ...')
    # trainer.train(num_epochs, train_dataset, val_dataset)
    
    