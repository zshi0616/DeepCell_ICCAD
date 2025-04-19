from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
from deepcell.dg_model import Model as DGModel
from deepcell.dg_trainer import Trainer as DGTrainer

from deepcell.dg_model import Model as DeepGate
from deepcell.dg3_model import Model as DeepGate3
from deepcell.pg_model import PolarGate
from deepcell.gcn_model import DirectMultiGCNEncoder as GCN

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_sample'
# checkpoint = './ckpt/aig_gcn.pth'

if __name__ == '__main__':
    args = get_parse_args()
    
    print('[INFO] Create Model and Trainer')
    if args.aig_encoder == 'pg':
        model = PolarGate(args, in_dim=3, out_dim=args.dim_hidden)
    elif args.aig_encoder == 'dg2':
        model = DeepGate(dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'dg3':
        model = DeepGate3(dim_hidden=args.dim_hidden)
    elif args.aig_encoder == 'gcn':
        model = GCN(dim_feature=3, dim_hidden=args.dim_hidden)
    
    # model.load(checkpoint)
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, random_sample=0.1)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Dataset Size Train: {:} / Test: {:}'.format(len(train_dataset), len(val_dataset)))
    
    trainer = DGTrainer(args, model, distributed=args.distributed, training_id=args.exp_id, device=args.device)
    if args.resume:
        trainer.resume()
    # trainer.set_training_args(loss_weight=[1.0, 0.0, 1.0], lr=1e-4, lr_step=80)
    # print('[INFO] Stage 1 Training ...')
    # trainer.train(10, train_dataset, val_dataset)
    
    trainer.set_training_args(loss_weight=[1.0, 0.0, 0.0], lr=1e-4, lr_step=20)
    print('[INFO] Stage 1 Training ...')
    trainer.train(args.num_epochs, train_dataset, val_dataset)
    
    
    