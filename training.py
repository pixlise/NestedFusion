
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import datetime
import sys, argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import json
import pickle
from scipy.signal import savgol_filter
import itertools


np.random.seed(42)

from formats import experiment_pb2
from formats import  quantification_pb2

from skimage import io
import pandas as pd
import utils


from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, ChainDataset

import gc
import os
import pyro
import pyro.distributions as dist
import pyro.poutine
from pyro.infer import MCMC, NUTS
import math
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide.guides import AutoDiagonalNormal
import pyro.distributions.constraints as constraints
from tqdm import trange

import utils
from models import FusionModel, NaiveFusionModel, SeparableVAE, JointVAE


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

local_radius_px=5




def train_model(model_class, model_args, dataset, batch_size=512, num_epochs=1, learning_rate=1e-3, device=device):
    
    z_dim=model_args['latent_dim']
    c_dim=model_args['n_classes']
    h_dim=model_args['hidden_dim']
    h_depth=model_args['hidden_depth']

    pyro.clear_param_store()
    torch.cuda.empty_cache()
    gc.collect()


    model = model_class(**model_args)
        
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True)

    model.to(device)


    scheduler = pyro.optim.OneCycleLR({
                            'optimizer':torch.optim.Adam,
                            'optim_args':{
                                    'lr': learning_rate,
                                    'eps' : 1e-3,
                                    'weight_decay':1e-4,
                                },
                            'max_lr' : learning_rate,
                            'epochs' : num_epochs,
                            'steps_per_epoch': int((1+len(dataset)//batch_size)),
                            'pct_start' : 0.25, 
                            'div_factor' : 1e3
                            })
    svi = pyro.infer.SVI(model  = model.model,
                        guide   = model.guide,
                        optim   = scheduler,
                        loss    = pyro.infer.Trace_ELBO())
    
    losses = []
    
    model_label = f'{model._get_name()}-z-{z_dim}-c-{c_dim}-h-{h_dim}-d-{h_depth}-r-{local_radius_px}'            
    model_params = sum(p.numel() for p in model.parameters())

    with tqdm(total=int(num_epochs*((len(dataset)//batch_size)+1)),position=0,leave=True,desc='Loss: Initializing..') as pbar:
        pbar.clear()
        for epoch in range(num_epochs):
            current_lr = learning_rate
            for x,q in data_loader:
                x = x.to(device)
                q = q.to(device)
                loss = svi.step(x,q)/x.shape[0]
                losses.append(loss)
                epoch_loss = np.mean(losses[-(len(dataset)//batch_size):])
                
                current_lr = [v for v in scheduler.optim_objs.values()][0].get_last_lr()[0]
                scheduler.step()

                pbar.set_description(f'Epoch: {epoch}  LR: {current_lr:0.2E}    Epoch Loss: {epoch_loss:.3E}    Batch Loss: {loss:.3E}  ')
                pbar.update()


            torch.save(model,f'models/{model_label}.pt')

            dbfile = open(f'models/{model_label}-losses.pkl', 'wb')
            pickle.dump(losses,dbfile)
            dbfile.close() 

def main(z_values, c_values, model_type, epochs, learning_rate=1e-4, batch_size=64,h=64,d=8):
    import data
    dataset = data.get_dataset(local_radius_px=local_radius_px)
  

    for z,c in tqdm(list(itertools.product(z_values,c_values))):
        print(f'z: {z}    c: {c}')
        train_model(model_type, 
            {
                'latent_dim'    : z,
                'n_classes'     : c,
                'rgbu_dim'      : 16,
                'rgbu_samples'  : (2*local_radius_px)**2,
                'quant_dim'     : 52,
                'hidden_dim'    : h,
                'hidden_depth'  : d,
                'dropout'       : 0.0,
                'skip_connection': True,
                'device'        : device,
                'x_err_gamma'   : 0.1,
                'q_err_gamma'   : 0.325,
            },
            **{
                'num_epochs'    : epochs,
                'learning_rate' : learning_rate,
                'batch_size'    : batch_size,
                'dataset'       : dataset,
                'device'        : device
            })



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Train PixlDB XRF and MCC Data.")
    parser.add_argument('-z', metavar='Z', type=int, nargs='+', required=True, help='Latent Dimensions')
    parser.add_argument('-c', metavar='C', type=int, nargs='+', required=True, help='Latent Class Numbers')
    parser.add_argument('-m', metavar='M', type=str, choices=['NaiveFusionModel', 'FusionModel', 'SeparableVAE', 'JointVAE'], required=True, help='Model Class')
    parser.add_argument('--batch-size', metavar='Batch Size', type=int, required=True, help='Batch Size (Rec 64)')
    parser.add_argument('--epochs', metavar='Epochs', type=int, required=True, help='Number of Epochs to Train')
    parser.add_argument('--lr', metavar='Learning Rate', type=float, required=True, help='Learning Rate (Rec 2e-3)')
    parser.add_argument('--hidden', metavar='Hidden Dimension', type=int, required=True, help='Hidden Dimension (Rec 64)')
    parser.add_argument('--depth', metavar='Hidden Depth ', type=int, required=True, help='HIdden Depth (Rec 8)')

    args = parser.parse_args()
    main_args = {
        'z_values' : args.z,
        'c_values' : args.c,
        'model_type' : eval(args.m),
        'epochs' :  args.epochs,
        'batch_size' : args.batch_size,
        'learning_rate' : args.lr,
        'h':args.hidden,
        'd':args.depth
    }

    main(**main_args)    
