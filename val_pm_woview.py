from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
from torch import nn
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_test'

if __name__ == '__main__':
    args = get_parse_args()
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, trainval_split=0, random_shuffle=False)
    _, val_dataset = dataset.get_dataset()
    
    print('[INFO] Create Model')
    model = deepcell.Model(aggr=args.pm_aggr).to(args.device)
    # checkpoint = './ckpt/pm_{}.pth'.format(args.pm_aggr)
    checkpoint = '/uac/gds/zyshi21/studio/DeepCell_ICCAD/exp/top_pm_gcn_dg2/top_pm_gcn_dg2/model_last.pth'
    model.load(checkpoint)
    model.eval()
    print('[INFO] Load checkpoint from {}'.format(checkpoint))
    
    # Test
    start_time = time.time()
    reg_loss = nn.L1Loss().to(args.device)
    prob_loss_list = []
    for g in val_dataset:
        hs, hf = model(g.to(args.device))
        prob = model.pred_prob(hf)
        prob_loss = reg_loss(prob, g['prob'].unsqueeze(1))
        prob_loss_list.append(prob_loss.item())
        print('[{:} / {:}] {} - Loss: {:.4f} - Time: {:.2f}s - ETA: {:.2f}s'.format(
            len(prob_loss_list), len(val_dataset), g['name'], prob_loss.item(), time.time() - start_time,
            (time.time() - start_time) / len(prob_loss_list) * (len(val_dataset) - len(prob_loss_list))
        ))
    print('{} | Test Loss: {:.4f}'.format(args.pm_aggr, sum(prob_loss_list) / len(prob_loss_list)))
    print()
    