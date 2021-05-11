import os
import sys
import time
import numpy as np
import datetime
import pickle as pkl
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pdb
from tqdm import tqdm
from datetime import datetime
import shutil
import torch
import logging
import json
import time

class ModelTrainer:

    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, config, loss_type, dataset_name, epochs, device, save_path, ckpt_path=None, comment=None):
        

        # Essential parts
        self.device = torch.device('cuda:{}'.format(device))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sch_name = config['scheduler']['name']
        
        self.loss_type = loss_type
        self.exp_path = Path(os.path.join(save_path, dataset_name, datetime.now().strftime('%d%B_%0l%0M'))) #21November_0430
        self.exp_path.mkdir(exist_ok=True, parents=True)

        
        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_path, 'training.log'))
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        
        #Dump hyper-parameters
        # config_info = {'optim':str(self.optimizer), 'scheduler':str(self.scheduler), 'criterion':str(self.criterion)}
        with open(str(self.exp_path.joinpath('config.json')), 'w') as f:
            json.dump(config, f, indent=2)

        if comment != None:
            self.logger.info(comment)

        
        self.dataset_name = dataset_name
        self.writter = SummaryWriter(self.exp_path.joinpath('logs'))
        self.epochs = epochs
        self.best_acc = 0.0
        self.best_epoch = 0
        
        self.flag = 0
        
        if ckpt_path != None:
            self.load_checkpoint(ckpt_path)


    def train(self):
        
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            train_loss, t_accuracy= self.train_single_epoch(epoch)
            valid_loss, v_accuracy = self.inference(epoch)
            duration = time.time() - start

            if v_accuracy > self.best_acc:
                self.best_acc = v_accuracy
                self.best_epoch = epoch
                self.flag = 1

            if self.sch_name == 'multistep':
                self.scheduler.step()
            elif self.sch_name == 'plateau':
                self.scheduler.step(valid_loss)

            self.logger.info("epoch: {} --- t_loss : {:0.3f}, train_acc = {}%, v_loss: {:0.3f}, val_acc: {}%, best_acc: {}%, best_epoch: {}, time: {:0.2f}s"\
                                                            .format(epoch, train_loss, t_accuracy, valid_loss, v_accuracy, self.best_acc, self.best_epoch, duration))
            
            
            if self.flag == 1:
                self.save_checkpoint(epoch, v_accuracy, True)
                self.flag = 0

            else:
                self.save_checkpoint(epoch, v_accuracy, False)

            self.writter.add_scalar('data/Train_Loss', train_loss, epoch)
            self.writter.add_scalar('data/Valid_Loss', valid_loss, epoch)
            self.writter.add_scalar('data/Train_Accuracy', t_accuracy, epoch)
            self.writter.add_scalar('data/Valid_Accuracy', v_accuracy, epoch)
        self.writter.close()


    def train_single_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0

        batch_size = len(self.train_loader)
        for b, batch in enumerate(self.train_loader):

            images, labels = batch            
            B, T, C, H, W = images.shape
            
            self.optimizer.zero_grad()
            
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs  = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()
            _, argmax = outputs.max(1)
            
            correct_cnt += (argmax == labels).sum()
            tot_cnt += B
                    
            
            print("{}/{} --- {}".format(b, batch_size, batch_loss), end='\r')

        return total_loss, (correct_cnt.item()/tot_cnt)*100


    def inference(self, epoch):
        
        self.model.eval()
        total_loss = 0.0
        accuracy = 0.0
        correct_cnt = 0
        tot_cnt = 0

        with torch.no_grad():
            for b, batch in enumerate(self.valid_loader):
                images, labels = batch
                B, T, C, H, W = images.shape
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs  = self.model(images)
                batch_loss = self.criterion(outputs, labels)

                total_loss += batch_loss.item()

                _, argmax = outputs.max(1)
                correct_cnt += (argmax == labels).sum()
                tot_cnt += B
                
        return total_loss, (correct_cnt.item()/tot_cnt)*100


    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        checkpoint = torch.load(ckpt, map_location=self.device)
        pretrained_dict = checkpoint['model_state_dict']
        optimizer_params = checkpoint['optimizer']
        self.model.load_state_dict(pretrained_dict)
        self.optimizer.load_state_dict(optimizer_params)

    def save_checkpoint(self, epoch, vacc, best=True):
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        self.exp_path.joinpath('ckpt').mkdir(exist_ok=True, parents=True)
        save_path = "{}/ckpt/last.pt".format(self.exp_path)
        torch.save(state_dict, save_path)
        if best:
            shutil.copyfile(save_path, '{}/ckpt/best.pt'.format(self.exp_path))
