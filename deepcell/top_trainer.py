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
from .utils.utils import zero_normalization, AverageMeter, get_function_acc, generate_k_hop_tensor
from .utils.logger import Logger

class TopTrainer():
    def __init__(self,
                 args, 
                 model, 
                 loss_weight = [1.0, 1.0, 1.0], 
                 device = 'cpu', 
                 distributed = True
                 ):
        super(TopTrainer, self).__init__()
        # Config
        self.args = args
        self.emb_dim = args.dim_hidden
        self.device = device
        self.lr = args.lr
        self.lr_step = args.lr_step
        self.loss_weight = loss_weight
        training_id = args.exp_id
        # Log Path
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        
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
        self.model_epoch = 0
        
        # Logger
        self.log_dir = os.path.join(args.save_dir, training_id)
        if self.local_rank == 0:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.log_path = os.path.join(self.log_dir, 'log-{}.txt'.format(time_str))
            self.logger = Logger(self.log_path)
            self.logger.write(str(vars(self.args)))
            print('[INFO] Create log file: {}'.format(self.log_path))
        
    def set_training_args(self, loss_weight=[], lr=-1, lr_step=-1, device='null'):
        if len(loss_weight) == 3 and loss_weight != self.loss_weight:
            print('[INFO] Update loss weight from {} to {}'.format(self.loss_weight, loss_weight))
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
            self.model.load(model_path)
            print('[INFO] Resume training from epoch {:}'.format(self.model_epoch))
            # self.load(model_path)
            return True
        else:
            raise ValueError('No model to resume')
        
    def run_batch(self, batch):
        mask_indices, mcm_tokens, gt_tokens, prob, watch_prob = self.model(batch)
        
        # Task 1: Probability Prediction 
        if self.args.refine == 'aig':
            prob_loss = self.reg_loss(prob[batch.aig_gate == 1], batch['aig_prob'].unsqueeze(1)[batch.aig_gate == 1])
            watch_prob_loss = self.reg_loss(watch_prob, batch['prob'].unsqueeze(1))
        else:
            prob_loss = self.reg_loss(prob, batch['prob'].unsqueeze(1))
            watch_prob_loss = self.reg_loss(watch_prob[batch.aig_gate == 1], batch['aig_prob'].unsqueeze(1)[batch.aig_gate == 1])
            
        # Task 2: Mask PM Circuit Modeling  
        mcm_loss = self.reg_loss(mcm_tokens[mask_indices], gt_tokens[mask_indices])
        
        loss_status = {
            'prob_loss': prob_loss,
            'mcm_loss': mcm_loss, 
            'watch': watch_prob_loss
        }
        return loss_status
    
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
            if train_dataset != None:
                train_dataset = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
            if val_dataset != None:
                val_dataset = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers)
        
        if self.args.aig_encoder == 'hoga':
            new_train_dataset = []
            new_val_dataset = []
            start_time = time.time()
            for batch_id, batch in enumerate(train_dataset):
                g = generate_k_hop_tensor(batch, 5)
                new_train_dataset.append(g)
                if self.local_rank == 0:
                    print('Process dataset for HOGA: [{} / {}] ETA: {:.2f}s'.format(
                        batch_id, len(train_dataset), (time.time() - start_time) / (batch_id + 1) * (len(train_dataset) - batch_id - 1)
                    ))
            for batch_id, batch in enumerate(val_dataset):
                g = generate_k_hop_tensor(batch, 5)
                new_val_dataset.append(g)
            train_dataset = new_train_dataset
            val_dataset = new_val_dataset
        
        
        # AverageMeter
        batch_time = AverageMeter()
        prob_loss_stats, mcm_loss_stats = AverageMeter(), AverageMeter()
        watch_loss_stats = AverageMeter()
        
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
                if dataset == None:
                    continue
                if self.local_rank == 0:
                    bar = Bar('{} {:}/{:}'.format(phase, epoch, num_epoch), max=len(dataset))
                
                prob_loss_stats.reset()
                mcm_loss_stats.reset()
                watch_loss_stats.reset()
                batch_time.reset()
                
                for iter_id, batch in enumerate(dataset):
                    batch = batch.to(self.device)
                    time_stamp = time.time()
                    # Get loss
                    loss_status = self.run_batch(batch)
                    loss = loss_status['prob_loss'] * self.loss_weight[0] + \
                        loss_status['mcm_loss'] * self.loss_weight[1]
                    loss /= sum(self.loss_weight)
                    loss = loss.mean()
                    
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    # Print and save log
                    batch_time.update(time.time() - time_stamp)
                    prob_loss_stats.update(loss_status['prob_loss'].item())
                    mcm_loss_stats.update(loss_status['mcm_loss'].item())
                    watch_loss_stats.update(loss_status['watch'].item())
                    if self.local_rank == 0:
                        Bar.suffix = '[{:}/{:}] [{}] |Tot: {total:} |ETA: {eta:} '.format(iter_id, len(dataset), self.args.refine, total=bar.elapsed_td, eta=bar.eta_td)
                        Bar.suffix += '|MCM: {:.4f} |Prob: {:.4f} |Watch: {:.4f} '.format(mcm_loss_stats.avg, prob_loss_stats.avg, watch_loss_stats.avg)
                        Bar.suffix += '|Net: {:.2f}s '.format(batch_time.avg)
                        bar.next()
                if self.local_rank == 0 and phase == 'train':
                    self.save(os.path.join(self.log_dir, 'model_last.pth'))
                    if self.model_epoch % 10 == 0:
                        self.save(os.path.join(self.log_dir, 'model_{:}.pth'.format(self.model_epoch)))
                if self.local_rank == 0:
                    self.logger.write('{}| Epoch: {:}/{:} |MCM: {:.4f} |Prob: {:.4f} |Net: {:.2f}s\n'.format(
                        phase, epoch, num_epoch, mcm_loss_stats.avg, prob_loss_stats.avg, batch_time.avg))
                    bar.finish()
            
            # Learning rate decay
            self.model_epoch += 1
            if self.lr_step > 0 and self.model_epoch % self.lr_step == 0:
                self.lr *= 0.1
                if self.local_rank == 0:
                    print('[INFO] Learning rate decay to {}'.format(self.lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
            
