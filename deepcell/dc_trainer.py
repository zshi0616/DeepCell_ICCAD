from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import time
from progress.bar import Bar
from torch_geometric.loader import DataLoader

from .arch.mlp import MLP
from .utils.utils import zero_normalization, AverageMeter, get_function_acc
from .utils.logger import Logger

class Trainer():
    def __init__(self,
                 args, 
                 model, 
                 training_id = 'default',
                 save_dir = './exp', 
                 lr = 1e-4,
                 loss_weight = [3.0, 1.0, 2.0],
                 emb_dim = 128, 
                 device = 'cpu', 
                 distributed = False
                 ):
        super(Trainer, self).__init__()
        # Config
        self.args = args
        self.emb_dim = emb_dim
        self.device = device
        self.lr = lr
        self.lr_step = -1
        self.loss_weight = loss_weight
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log_dir = os.path.join(save_dir, training_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.distributed = distributed and torch.cuda.is_available()
        
        # Distributed Training 
        self.local_rank = 0
        if self.distributed:
            if 'LOCAL_RANK' in os.environ:
                self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = 'cuda:%d' % args.gpus[self.local_rank]
            torch.cuda.set_device(args.gpus[self.local_rank])
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            print('Training in distributed mode. Device {}, Process {:}, total {:}.'.format(
                self.device, self.rank, self.world_size
            ))
        else:
            print('Training in single device: ', self.device)
        
        # Loss and Optimizer
        self.reg_loss = nn.L1Loss().to(self.device)
        self.clf_loss = nn.BCELoss().to(self.device)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Model
        self.model = model.to(self.device)
        self.readout_rc = MLP(emb_dim * 2, 32, 1, num_layer=3, p_drop=0.1, norm_layer='batchnorm').to(self.device)
        self.model_epoch = 0
        
        # Logger
        if self.local_rank == 0:
            self.logger = Logger(self.log_path)
        
    def set_training_args(self, loss_weight=[], lr=-1, lr_step=-1, device='null'):
        if len(loss_weight) == 3 and loss_weight != self.loss_weight:
            print('[INFO] Update loss_weight from {} to {}'.format(self.loss_weight, loss_weight))
            self.loss_weight = loss_weight
        if lr > 0 and lr != self.lr:
            print('[INFO] Update learning rate from {} to {}'.format(self.lr, lr))
            self.lr = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        if lr_step > 0 and lr_step != self.lr_step:
            print('[INFO] Update learning rate step from {} to {}'.format(self.lr_step, lr_step))
            self.lr_step = lr_step
        if device != 'null' and device != self.device:
            print('[INFO] Update device from {} to {}'.format(self.device, device))
            self.device = device
            self.model = self.model.to(self.device)
            self.reg_loss = self.reg_loss.to(self.device)
            self.clf_loss = self.clf_loss.to(self.device)
            self.optimizer = self.optimizer
            self.readout_rc = self.readout_rc.to(self.device)
        
    def save(self, path):
        data = {
            'epoch': self.model_epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(data, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            self.lr = param_group['lr']
        self.model_epoch = checkpoint['epoch']
        self.model.load(path)
        print('[INFO] Continue training from epoch {:}'.format(self.model_epoch))
        return path
    
    def resume(self):
        model_path = os.path.join(self.log_dir, 'model_last.pth')
        if os.path.exists(model_path):
            self.load(model_path)
            return True
        else:
            return False
        
    def run_batch(self, batch):
        hs, hf = self.model(batch)
        prob = self.model.pred_prob(hf)
        connect = self.model.pred_connect(batch, hs)
        # Task 1: Probability Prediction 
        prob_loss = self.reg_loss(prob, batch['prob'].unsqueeze(1))
        # Task 2: Functional Similarity 
        if len(batch['tt_dis']) != 0:
            node_a = hf[batch['tt_pair_index'][0]]
            node_b = hf[batch['tt_pair_index'][1]]
            emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
            emb_dis_z = zero_normalization(emb_dis)
            tt_dis_z = zero_normalization(batch['tt_dis'])
            func_loss = self.reg_loss(emb_dis_z, tt_dis_z)
        else:
            func_loss = torch.tensor(0.0).to(self.device)
        # Task 3: Structural Prediction
        if len(batch['connect_label']) != 0:
            con_loss = self.ce_loss(connect, batch['connect_label'].long())
        else:
            con_loss = torch.tensor(0.0).to(self.device)

        loss_status = {
            'prob_loss': prob_loss, 
            'func_loss': func_loss, 
            'con_loss': con_loss, 
        }
        
        if torch.isnan(func_loss).any(): 
            loss_status['func_loss'] = torch.tensor(0.0).to(self.device)
        if torch.isnan(con_loss).any(): 
            loss_status['con_loss'] = torch.tensor(0.0).to(self.device)
        
        return hs, hf, loss_status
    
    def train(self, num_epoch, train_dataset, val_dataset):
        # Distribute Dataset
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                    num_workers=self.num_workers, sampler=train_sampler)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True,
                                     num_workers=self.num_workers, sampler=val_sampler)
        else:
            train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        # AverageMeter
        batch_time = AverageMeter()
        prob_loss_stats, func_loss_stats, con_loss_stats = AverageMeter(), AverageMeter(), AverageMeter()
        acc_stats = AverageMeter()
        
        # Train
        print('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        for epoch in range(num_epoch): 
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataset = train_dataset
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataset = val_dataset
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    hs, hf, loss_status = self.run_batch(batch)

                    # loss = loss_status['prob_loss'] * self.loss_weight[0] + \
                    #     loss_status['rc_loss'] * self.loss_weight[1] + \
                    #     loss_status['func_loss'] * self.loss_weight[2]
                    
                    loss = loss_status['prob_loss'] * self.loss_weight[0] + \
                        loss_status['func_loss'] * self.loss_weight[1] + \
                        loss_status['con_loss'] * self.loss_weight[2]
                    
                    loss /= sum(self.loss_weight)
                    loss = loss.mean()
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)
                    prob_loss_stats.update(loss_status['prob_loss'].item())
                    func_loss_stats.update(loss_status['func_loss'].item())
                    con_loss_stats.update(loss_status['con_loss'].item())
                    # acc = get_function_acc(batch, hf)
                    # acc = 0
                    # acc_stats.update(acc)
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}]|Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), total=bar.elapsed_td, eta=bar.eta_td)
                        # Bar.suffix += '|Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} '.format(prob_loss_stats.avg, con_loss_stats.avg, func_loss_stats.avg)
                        Bar.suffix += '|Prob: {:.4f} |Func: {:.4f} |Con: {:.4f} '.format(prob_loss_stats.avg, func_loss_stats.avg, con_loss_stats.avg)
                        # Bar.suffix += '|Acc: {:.2f}%% '.format(acc*100)
                        Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                        bar.next()
                if phase == 'train':
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                    if self.model_epoch % 10 == 0:
                        self.save(os.path.join(self.log_dir, 'model_{:}.pth'.format(self.model_epoch)))
                if self.local_rank == 0:
                    # self.logger.write('{}| Epoch: {:}/{:} |Prob: {:.4f} |RC: {:.4f} |Func: {:.4f} |ACC: {:.4f} |Net: {:.2f}s\n'.format(
                    #     phase, epoch, num_epoch, prob_loss_stats.avg, con_loss_stats.avg, func_loss_stats.avg, acc_stats.avg, batch_time.avg))
                    self.logger.write('{}| Epoch: {:}/{:} |Prob: {:.4f} |Func: {:.4f} |Net: {:.2f}s\n'.format(
                        phase, epoch, num_epoch, prob_loss_stats.avg, func_loss_stats.avg, batch_time.avg))
                    bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            