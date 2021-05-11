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
        self.model.eval()
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
                _, output = torch.max(outputs, 1)

                if output.item() == 1:
                    output = 1
                else:
                    output = 0

                output_list.append(output)

        return output_list

    def visualizaition(self):
        # to be updated
        pass

