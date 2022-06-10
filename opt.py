import json
import yaml
import argparse
import numpy as np

import torch

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from architect import AE, VAE
from losses import Parallel_Loss, VAE_Loss
from models import ParallelAE, Variational


RANDOM_SEED = 1412912
rng = np.random.default_rng(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
    
def run_hyperopt(hyperopt_config=None):
    # Extract options (Part 01)
    config = hyperopt_config['suggestion']
    
    config['batch'] = int(2**config['batch'])
    config['kl_scaling'] = 10**config['kl_scaling']
    config['lr'] = 10**config['lr']
    config['scheduler_step'] = int(config['epochs']/config['scheduler_step'])
    
    config['variational'] = True
    config['name'] = 'vae'
    config['model'] = VAE(emb_dim=config['emb_dim']) if config['variational'] else AE(emb_dim=config['emb_dim'])
	config['loss'] = VAE_Loss(config) if config['variational'] else Parallel_Loss(config)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and split dataset (Part 02)
    trainset = MNIST(
                    root='./', 
                    train=True, 
                    transform=Compose([
                        ToTensor(),
                        Normalize((0.1307,), (0.3081,))
                        ]), 
                    download=True)		
	
    test_data = MNIST(  
                    root='./', 
                    train=False, 
                    transform=Compose([
                        ToTensor(),
                        Normalize((0.1307,), (0.3081,))
                        ]), 
                    download=True)	                 
    train_data, val_data = torch.utils.data.random_split(trainset, [50000, 10000])
    
    # train and evaluate model (Part 03)
    model = Variational(config) if config['variational'] else ParallelAE(config)
    train_log = model.fit(train_data, val_data)  
    result = model.test(test_data)['full']

    return result, train_log
    
    
    
    
    
    
    
    
    
        

