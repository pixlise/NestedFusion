from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import json
import pickle


np.random.seed(42)

from formats import experiment_pb2
from formats import  quantification_pb2

from skimage import io
import pandas as pd
import utils

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, ChainDataset


import os
import pyro
import pyro.distributions as dist
import pyro.poutine
from pyro.infer import MCMC, NUTS
import math
import torch.nn as nn
import torch.nn.functional as F

from tqdm import trange
from IPython.display import clear_output



dataset_df = utils.load_dataset_dataframe()
quant_df = utils.load_quant_dataframe()

quant_keys = set()
for quant in tqdm(quant_df.quant, leave=False):
    quant_keys = quant_keys.union([k for k in quant.keys() if not '/' in k])

quant_keys = sorted(quant_keys)

def quant_dict_to_vec(d):
    vec = np.zeros(len(quant_keys))
    for k in d:
        if k in quant_keys:
            vec[quant_keys.index(k)] = d[k]
    return vec

def get_ratios_img(dataset, standardize=True):
    data = utils.load_rgbu_array(dataset)
    primal_channels = data.shape[2]
    for channel_i in range(primal_channels):
        for channel_j in range(primal_channels):
            if channel_i != channel_j:
                ratio = data[:,:,channel_i]/(data[:,:,channel_j]+1)
                ratio = ratio.reshape(ratio.shape[0],ratio.shape[1],1)
                data = np.concatenate((data,ratio),-1)
    if standardize:
        ratio_channels = data.shape[2]
        for channel in range(ratio_channels):
            data[:,:,channel] /= data[:,:,channel].std() 
    return data

class PixlMap(Dataset):
    def __init__(self,dataset,local_radius_px=5,standardize=False):
        super().__init__()
        self.rgbu_data = get_ratios_img(dataset,standardize=standardize)
        self.pmc_map = utils.get_pmc_rgbu_map(dataset)
        self.quant_data = quant_df[quant_df.dataset == dataset]
        self.pmc = list(self.quant_data.pmc.unique())
        self.local_radius = local_radius_px
        self.rgbu_samples = (2*self.local_radius)**2
        self.rgbu_dim = self.rgbu_data.shape[-1]
        self.quant_dim = quant_dict_to_vec(self.quant_data.quant.iloc[0]).shape[0]
            
    def __getitem__(self, i):
        pmc = self.pmc[i]
        quant_vec = quant_dict_to_vec(self.quant_data[self.quant_data.pmc==pmc].quant.iloc[0])
        point = self.pmc_map[pmc]
        local_rgbu = self.rgbu_data[int(point['j'])-self.local_radius:int(point['j'])+self.local_radius,int(point['i'])-self.local_radius:int(point['i'])+self.local_radius,:]
        rgbu_vec = local_rgbu.reshape(-1,self.rgbu_dim)
        if rgbu_vec.shape[0]<self.rgbu_samples:
            extra_empty_vecs = self.rgbu_samples-rgbu_vec.shape[0]
            rgbu_vec = np.vstack((rgbu_vec,np.zeros((extra_empty_vecs,self.rgbu_dim))))
            
        
        return rgbu_vec.astype(np.float32), quant_vec.astype(np.float32)

    def __len__(self):
        return len(self.pmc)

    def convert_to_model_batch(self):
        x_batch, q_batch = [], []
        for i in range(len(self)):
            x, q = self[i]
            x_batch.append(x)
            q_batch.append(q)

        x_batch = np.array(x_batch)
        q_batch = np.array(q_batch)
        return x_batch , q_batch

    def convert_batch_latents_to_image(self, z):
        img_dict = dict()
        for i in range(len(z)):
            patch = z[i].reshape(2*self.local_radius,2*self.local_radius,-1)
                    
            pmc = self.pmc[i]
            point = self.pmc_map[pmc]
            for img_j in range(int(point['j'])-self.local_radius,int(point['j'])+self.local_radius):
                for img_i in range(int(point['i'])-self.local_radius,int(point['i'])+self.local_radius):
                    latent = patch[img_j-(int(point['j'])-self.local_radius),img_i-(int(point['i'])-self.local_radius)]
                    if (img_j,img_i) in img_dict:
                        img_dict[(img_j,img_i)].append(latent)
                    else:
                        img_dict[(img_j,img_i)] = [latent]

        img = np.zeros((self.rgbu_data.shape[0],self.rgbu_data.shape[1],z.shape[-1]))
        for (img_j,img_i) in img_dict:
            overlapping_patch_set = np.array(img_dict[(img_j,img_i)]).reshape((len(img_dict[(img_j,img_i)]),-1))
            
            img[img_j,img_i,:] = np.mean(overlapping_patch_set,axis=0)
        return img

def get_dataset(local_radius_px=5,standardize_rgbu=False):
    ds_kwargs = {
        'local_radius_px':local_radius_px,
        'standardize':standardize_rgbu
    }
    # dataset = PixlMap('208601602',**ds_kwargs)
    dataset_list=[ds for ds in tqdm(quant_df.dataset.unique(),leave=False) if (utils.check_rgbu(ds))]
    dataset = ConcatDataset([PixlMap(ds,**ds_kwargs) for ds in tqdm(dataset_list,leave=False)])
    return dataset


clear_output(wait=True)