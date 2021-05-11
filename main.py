import sys
sys.path.append('./')
import argparse
import pdb
import argparse
import yaml 
from torch.utils.data import DataLoader
from trainer import ModelTrainer
from tester import ModelTester
from utils.setup import setup_solver
import os
import pickle
from model import MYNET

def train(args):
    with open(os.path.join(args.config,args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    model = MYNET(**config['MYNET'])
    
    if args.dataset == 'RWF':
        from datasets.RWF import RWF
        train_dataset = RWF(config['datasets']['train'], 'train', config['MYNET']['sequence_size'], temporal_stride=config['datasets']['stride'])
        valid_dataset = RWF(config['datasets']['valid'], 'valid', config['MYNET']['sequence_size'])
    
    train_loader = DataLoader(train_dataset, **config['dataloader']['train'])
    valid_loader = DataLoader(valid_dataset, **config['dataloader']['valid'])

    optimizer, scheduler, criterion = setup_solver(model.parameters(), config)
    
    trainer = ModelTrainer(model, train_loader, valid_loader, criterion, optimizer, scheduler, config, config['criterion']['name'], dataset_name=config['datasets']['name'], **config['trainer'])
    trainer.train()


def test(args):
    with open(os.path.join(args.config,args.dataset + '.yml'), mode='r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    model = MYNET(**config['MYNET'])
    
    if args.dataset == 'RWF':
        from datasets.RWF_test import RWF_test
        test_dataset = RWF_test(config['datasets']['test'], config['MYNET']['sequence_size'])

    test_loader = DataLoader(test_dataset, **config['dataloader']['test'])
    tester = ModelTester(model, test_loader, **config['tester'])
    output = tester.test()
    with open("output.pkl", "wb") as f:
        pickle.dump(output, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.', help='Root directory')
    parser.add_argument('-c', '--config', type=str, help='Path to option YAML file.')
    parser.add_argument('-d', '--dataset', type=str, help='Dataset')
    parser.add_argument('-m', '--mode', type=str, help='Train or Test')
    args = parser.parse_args()

    if args.mode == 'Train':
        train(args)
    elif args.mode == 'Test':
        test(args)
