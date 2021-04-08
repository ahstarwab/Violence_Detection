import os
import sys
import time
import numpy as np
import datetime

import pickle as pkl

from pathlib import Path
import cv2
import torch
import pdb
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F

import logging
import json
from multiprocessing import Pool
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

            self.scheduler.step(v_accuracy)

            if self.dataset_name == 'AI_HUB':
                self.logger.info("epoch: {} --- t_loss : {:0.3f}, train_acc = {}%, v_loss: {:0.3f}, val_acc: {}%, best_acc: {}%, best_epoch: {}, time: {:0.2f}s"\
                                                            .format(epoch, train_loss, t_accuracy/2, valid_loss, v_accuracy/2, self.best_acc, self.best_epoch, duration))
            else:
                self.logger.info("epoch: {} --- t_loss : {:0.3f}, train_acc = {}%, v_loss: {:0.3f}, val_acc: {}%, best_acc: {}%, best_epoch: {}, time: {:0.2f}s"\
                                                            .format(epoch, train_loss, t_accuracy, valid_loss, v_accuracy, self.best_acc, self.best_epoch, duration))
    
                    # self.logger.info("tvloss:{:0.3f}, constrast loss:{:0.3f}, my loss:{:0.3f}"\
                #                                             .format(losses[0].item(), losses[1].item(), losses[2].item()))

            self.save_checkpoint(epoch, v_accuracy)

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
            
            """
            images : [B x 2 x T x C x H x W]
            labels : [B x 2 x T]
            """
            # pdb.set_trace()
            images, labels = batch
            
            B, T, C, H, W = images.shape
            
            self.optimizer.zero_grad()
            
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs  = self.model(images)
            # outputs, losses = self.model(images)
            
            if self.loss_type == 'CrossEntropy':    
                batch_loss = self.criterion(outputs, labels)

            # 
            # print("{}, {}, {}, {}".format(batch_loss, losses[0].item(), losses[1].item(), losses[2]), end='\r')

            # if epoch < 20:
            # batch_loss += (losses[0] + losses[1] + losses[2])

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
                # outputs, losses = self.model(images)
                
                # if self.dataset_name == 'AI_HUB':
                #     labels = labels[:,-1]
                    
                batch_loss = self.criterion(outputs, labels)

                # if epoch < 20:
                # batch_loss += (losses[0] + losses[1] + losses[2])

                total_loss += batch_loss.item()

                _, argmax = outputs.max(1)
                correct_cnt += (argmax == labels).sum()
                tot_cnt += B
                
        return total_loss, (correct_cnt.item()/tot_cnt)*100


    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        # print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt, map_location=self.device)
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()if (k in model_dict and 'base_model.classifier' not in k)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        # self.optimizer.load_state_dict(checkpoint['optimizer'])

        # self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)


    def save_checkpoint(self, epoch, vacc, best=True):
        
        state_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        self.exp_path.joinpath('ckpt').mkdir(exist_ok=True, parents=True)
        save_path = "{}/ckpt/{}_{:0.4f}.pt".format(self.exp_path, epoch, vacc)
        torch.save(state_dict, save_path)


class ModelTester:
    def __init__(self, model, test_loader, ckpt_path, device):

        # Essential parts
        self.device = torch.device('cuda:{}'.format(device))
        self.model = model.to(self.device)
        self.test_loader = test_loader
        # Set logger
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(sh)

        self.load_checkpoint(ckpt_path)


    def load_checkpoint(self, ckpt):
        self.logger.info(f"Loading checkpoint from {ckpt}")
        # print('Loading checkpoint : {}'.format(ckpt))
        checkpoint = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        


    def test(self):
        """
        images : [B x T x C x H x W]
        labels : [B x T]
        """
        self.model.eval()
        total_loss = 0.0
        
        output_list = []
        label_list = []

        with torch.no_grad():
            for b, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                # [B x Class]
                
                _, output = torch.max(outputs, 1)
                
                output_list.extend(output)
                label_list.extend(labels)
                batch_acc = float(len(output) - sum(abs(output-labels)))/len(output)
                self.logger.info(f"Batch_Accuracy : {batch_acc}")
        
        output_list = torch.tensor(output_list)
        label_list = torch.tensor(label_list)
        tot_acc = float(len(output_list) - sum(abs(output_list-label_list)))/len(output_list)
        self.logger.info(f"Final Accuracy : {tot_acc}")
        # pdb.set_trace()
        return output_list

    def demo(self):
        """
        images : [B x T x C x H x W]
        labels : [B x T]
        """
        self.model.eval()
        total_loss = 0.0
        
        output_list = []
        
        with torch.no_grad():
            for b, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                images = batch
                images = images.to(self.device)
                outputs = self.model(images)
                # [B x Class]
                
                _, output = torch.max(outputs, 1)

                if output.item() == 1:
                    output = 1
                else:
                    output = 0

                output_list.append(output)
                # batch_loss = self.criterion(outputs, labels[:,-1].long())     
                # total_loss += batch_loss.item()
        
        return output_list

    def visualizaition(self):
        # to be updated
        pass

