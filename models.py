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


from sklearn.decomposition import PCA
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
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide.guides import AutoDiagonalNormal
import pyro.distributions.constraints as constraints
from tqdm import trange


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, hidden_depth=3, activation=F.softplus, output_activation=None, dropout=0.0, skip_connection=True, device=None):
        super().__init__()
        self.input = nn.Linear(input_dim,hidden_dim,device=device)
        self.fc_layers = [nn.Linear(hidden_dim, hidden_dim, device=device) for layer in range(hidden_depth)]
        self.output = nn.Linear(hidden_dim, output_dim, device=device)
        self.activation = activation
        self.output_activation = output_activation
        self.dropout = nn.Dropout(dropout) 
        self.skip_connection = skip_connection
        self.skip_layer = nn.Linear(input_dim+hidden_dim,hidden_dim,device=device)


        for i,layer in enumerate(self.fc_layers):
            self.add_module(f'fc_{i}',layer)

    def forward(self, x):
        h = self.input(x)
        for layer in self.fc_layers:
            h = self.dropout(self.activation(layer(h)))
        if self.skip_connection:
            h = self.dropout(self.activation(self.skip_layer(torch.cat((x, h), -1))))    
        o = self.output(h)
        if self.output_activation:
            o = self.output_activation(o)
        return o
    
class Transformer(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=4, num_heads=4, dropout=0.0, internal_activation=F.softplus, output_activation=None, skip_connection=True, max_seq_len=100, single_output=False,include_head_in_output=False,internal_shuffle_seq_after_head=False, device=None):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
         
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True,
                device=device)
        self.add_module(f'enc_layer',self.transformer_encoder_layer)

        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer,num_layers=num_layers)  
        self.add_module(f'transformer_enc',self.transformer)

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.add_module(f'output_layer',self.output_layer)

        self.output_activation = output_activation
        self.internal_actiavtion = internal_activation

        self.dropout = nn.Dropout(dropout)

        self.skip_connection = skip_connection
        self.skip_layer = nn.Linear(input_dim+hidden_dim,hidden_dim,device=device)


        self.single_output=single_output 
        self.internal_shuffle_seq_after_head = internal_shuffle_seq_after_head
        self.include_head_in_output = include_head_in_output



    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        h = self.internal_actiavtion(self.input_layer(x)).clone()

        if self.internal_shuffle_seq_after_head:
            sequence_shuffle_indicies = torch.randperm(h.shape[1]-1)
            h[:,1:,:] = h[:,1:,:].clone()[:,sequence_shuffle_indicies,:]

        h = self.dropout(h)
        h = self.transformer(h,mask=mask)
        
        if self.internal_shuffle_seq_after_head:
            sequence_shuffle_inverse_indicies = torch.zeros_like(sequence_shuffle_indicies)
            for i in range(sequence_shuffle_indicies.shape[0]):
                sequence_shuffle_inverse_indicies[sequence_shuffle_indicies[i]]=i
            h[:,1:,:] = h[:,1:,:].clone()[:,sequence_shuffle_inverse_indicies,:]

        if self.skip_connection:
            h = self.dropout(self.internal_actiavtion(self.skip_layer(torch.cat((x, h), -1))))    

        o = self.output_layer(h)


        if self.output_activation:
            o = self.output_activation(o)
        if self.single_output:
            return o.mean(1)
        if not self.include_head_in_output:
            o = o[:,1:,:]

        return o         


class FusionModel(nn.Module):
    def __init__(self, rgbu_dim, rgbu_samples, quant_dim, latent_dim, n_classes=10 ,x_err_gamma=1.0, q_err_gamma=1.0, hidden_dim=20, hidden_depth=3, dropout=0.0, skip_connection=True, device=None):
        super().__init__()
        self.rgbu_dim = rgbu_dim
        self.quant_dim = quant_dim
        self.latent_dim = latent_dim 
        self.n_classes = n_classes
        self.rgbu_samples = rgbu_samples
        self.rgbu_latent_dim = latent_dim
        self.quant_latent_dim = latent_dim*rgbu_samples
        self.patch_latent_dim = latent_dim*rgbu_samples
        self.device = device
        self.latent_eps=1e-3
        self.x_err_gamma = x_err_gamma
        self.q_err_gamma = q_err_gamma

        self.rgbu_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = rgbu_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.quant_decoder = Transformer(
            input_dim      = latent_dim+n_classes,
            output_dim     = quant_dim,
            hidden_dim     = hidden_dim,
            num_layers     = hidden_depth,
            num_heads      = hidden_dim//4,
            max_seq_len    = 1 + rgbu_samples,
            dropout        = dropout,
            output_activation = F.relu,
            single_output  = True,
            internal_shuffle_seq_after_head = False,
            device         = device
        )

        self.joint_encoder_mu = MLP(
            input_dim       = quant_dim+rgbu_dim*rgbu_samples,
            output_dim      = latent_dim*rgbu_samples,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.joint_encoder_sigma = MLP(
            input_dim       = quant_dim+rgbu_dim*rgbu_samples,
            output_dim      = latent_dim*rgbu_samples,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = torch.sigmoid,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
        
        self.joint_encoder_class = MLP(
            input_dim       = quant_dim+rgbu_dim*rgbu_samples,
            output_dim      = n_classes*rgbu_samples,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

              
    def to(self,device):
        self.device = device
        return super().to(device)

    def model(self, x, q):
        pyro.module("rgbu_decoder", self.rgbu_decoder)
        pyro.module("quant_decoder", self.quant_decoder)


        batch_size = x.shape[0]
        
        prior_loc = torch.zeros((batch_size, self.rgbu_samples, self.latent_dim),device=self.device)
        prior_scale = torch.ones((batch_size, self.rgbu_samples, self.latent_dim),device=self.device)
        z = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(3))
        
        alpha = torch.ones((batch_size, self.n_classes),device=self.device)
        class_dist = pyro.sample("c_dist", dist.Dirichlet(alpha).to_event(1))

        c_prior =  class_dist.repeat(self.rgbu_samples,1,1).swapaxes(0,1)
        c = pyro.sample("c", dist.OneHotCategoricalStraightThrough(logits=c_prior).to_event(2))
        
        combined_latents = torch.cat((c, z), -1)

        q_hat = self.quant_decoder(combined_latents)
        q_err = torch.ones_like(q_hat,device=self.device)*(1.0/self.rgbu_samples)*self.q_err_gamma
        pyro.sample(
            'quant',
            dist.Normal(q_hat,q_err).to_event(2),
            obs=q
        ) 


        z_nested_batch_agg = z.reshape((batch_size*self.rgbu_samples, self.latent_dim))
        c_nested_batch_agg = c.reshape((batch_size*self.rgbu_samples, self.n_classes))
        nested_batch_combined_latents = torch.cat((c_nested_batch_agg, z_nested_batch_agg), 1)

        x_hat_flat = self.rgbu_decoder(nested_batch_combined_latents)
        x_hat = x_hat_flat.reshape(x.shape)
        x_err = torch.ones_like(x,device=self.device)*self.x_err_gamma
        pyro.sample(
            'rgbu',
            dist.Normal(x_hat, x_err).to_event(3),
            obs=x
        ) 

    def guide(self, x, q):
        pyro.module("joint_encoder_mu", self.joint_encoder_mu)
        pyro.module("joint_encoder_sigma", self.joint_encoder_sigma)
        pyro.module("joint_encoder_class", self.joint_encoder_class)

        batch_size = x.shape[0]
        
        x_flat = x.reshape(x.shape[0],-1)
        batch_data_concat = torch.cat((q, x_flat), 1)

        z_mu_flat = self.joint_encoder_mu(batch_data_concat)
        z_sigma_flat = self.joint_encoder_sigma(batch_data_concat)
        c_dist_flat = self.joint_encoder_class(batch_data_concat)

        z_mu = z_mu_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))
        z_sigma = z_sigma_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))+self.latent_eps
        c_estimate = c_dist_flat.reshape(((batch_size, self.rgbu_samples, self.n_classes)))

        z = pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(3))
        c = pyro.sample("c", dist.OneHotCategoricalStraightThrough(logits=c_estimate).to_event(2))
        class_dist = pyro.sample("c_dist", dist.Dirichlet(c.mean(1)+self.latent_eps).to_event(1))


    def encode(self, x, q):
        batch_size = x.shape[0]
        
        x_flat = x.reshape(x.shape[0],-1)
        batch_data_concat = torch.cat((q, x_flat), 1)

        z_mu_flat = self.joint_encoder_mu(batch_data_concat)
        z_sigma_flat = self.joint_encoder_sigma(batch_data_concat)
        c_dist_flat = self.joint_encoder_class(batch_data_concat)

        z_mu = z_mu_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))
        z_sigma = z_sigma_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))+self.latent_eps
        c_dist = c_dist_flat.reshape(((batch_size, self.rgbu_samples, self.n_classes)))

        z = dist.Normal(z_mu, z_sigma).sample()
        c = dist.OneHotCategoricalStraightThrough(logits=c_dist).sample()

        return  c, z
        
    def decode(self, c, z):
        batch_size = z.shape[0]
        
        combined_latents = torch.cat((c, z), -1)
        q_hat = self.quant_decoder(combined_latents)


        z_nested_batch_agg = z.reshape((batch_size*self.rgbu_samples, self.latent_dim))
        c_nested_batch_agg = c.reshape((batch_size*self.rgbu_samples, self.n_classes))
        nested_batch_combined_latents = torch.cat((c_nested_batch_agg, z_nested_batch_agg), 1)

        x_hat_flat = self.rgbu_decoder(nested_batch_combined_latents)
        x_hat = x_hat_flat.reshape((batch_size,self.rgbu_samples, self.rgbu_dim))

        return x_hat, q_hat

    def reconstruct(self, x, q):
        c, z = self.encode(x,q)
        x_hat, q_hat = self.decode(c, z)
        return x_hat.detach(), q_hat.detach()



class SeparableVAE(nn.Module):
    def __init__(self, rgbu_dim, rgbu_samples, quant_dim, latent_dim, n_classes=10 ,x_err_gamma=1.0, q_err_gamma=1.0, hidden_dim=20, hidden_depth=3, dropout=0.0, skip_connection=True, device=None):
        super().__init__()
        self.rgbu_dim = rgbu_dim
        self.rgbu_samples = rgbu_samples

        self.quant_dim = quant_dim
        self.latent_dim = latent_dim 
        self.n_classes = n_classes
        self.latent_eps=1e-3
        self.x_err_gamma = x_err_gamma
        self.q_err_gamma = q_err_gamma

        self.rgbu_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = rgbu_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.quant_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = quant_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.rgbu_encoder_mu = MLP(
            input_dim       = rgbu_dim,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.rgbu_encoder_sigma = MLP(
            input_dim       = rgbu_dim,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = torch.sigmoid,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
        
        self.rgbu_encoder_class = MLP(
            input_dim       = rgbu_dim,
            output_dim      = n_classes,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)


        self.quant_encoder_mu = MLP(
            input_dim       = quant_dim,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.quant_encoder_sigma = MLP(
            input_dim       = quant_dim,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = torch.sigmoid,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
        
        self.quant_encoder_class = MLP(
            input_dim       = quant_dim,
            output_dim      = n_classes,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
              
    def to(self,device):
        self.device = device
        return super().to(device)

    def model(self, x, q):
        pyro.module("rgbu_decoder", self.rgbu_decoder)
        pyro.module("quant_decoder", self.quant_decoder)

        batch_size = x.shape[0]
        
        prior_loc = torch.zeros((batch_size, self.rgbu_samples, self.latent_dim),device=self.device)
        prior_scale = torch.ones((batch_size, self.rgbu_samples, self.latent_dim),device=self.device)
        z_x = pyro.sample("z_x", dist.Normal(prior_loc, prior_scale).to_event(3))
        
        class_prior = torch.ones((batch_size, self.rgbu_samples, self.n_classes),device=self.device)
        c_x = pyro.sample("c_x", dist.OneHotCategoricalStraightThrough(logits=class_prior).to_event(2))

        combined_x_latents = torch.cat((c_x, z_x), -1)

        x_hat = self.rgbu_decoder(combined_x_latents)
        x_err = torch.ones_like(x,device=self.device)*self.x_err_gamma
        pyro.sample(
            'rgbu',
            dist.Normal(x_hat, x_err).to_event(3),
            obs=x
        ) 

        prior_loc = torch.zeros((batch_size, self.latent_dim),device=self.device)
        prior_scale = torch.ones((batch_size, self.latent_dim),device=self.device)
        z_q = pyro.sample("z_q", dist.Normal(prior_loc, prior_scale).to_event(2))
        
        class_prior = torch.ones((batch_size, self.n_classes),device=self.device)
        c_q = pyro.sample("c_q", dist.OneHotCategoricalStraightThrough(logits=class_prior).to_event(1))

        combined_q_latents = torch.cat((c_q, z_q), -1)
        
        q_hat = self.quant_decoder(combined_q_latents)
        q_err = torch.ones_like(q_hat,device=self.device)*(1.0/self.rgbu_samples)*self.q_err_gamma
        pyro.sample(
            'quant',
            dist.Normal(q_hat,q_err).to_event(2),
            obs=q
        ) 




    def guide(self, x, q):
        pyro.module("rgbu_encoder_mu", self.rgbu_encoder_mu)
        pyro.module("rgbu_encoder_sigma", self.rgbu_encoder_sigma)
        pyro.module("rgbu_encoder_class", self.rgbu_encoder_class)
        pyro.module("quant_encoder_mu", self.quant_encoder_mu)
        pyro.module("quant_encoder_sigma", self.quant_encoder_sigma)
        pyro.module("quant_encoder_class", self.quant_encoder_class)

        batch_size = x.shape[0]
        z_x_mu = self.rgbu_encoder_mu(x)
        z_x_sigma = self.rgbu_encoder_sigma(x)
        z_x = pyro.sample("z_x", dist.Normal(z_x_mu, z_x_sigma).to_event(3))
        
        c_x_dist = self.rgbu_encoder_class(x)
        c_x = pyro.sample("c_x", dist.OneHotCategoricalStraightThrough(logits=c_x_dist).to_event(2))

        z_q_mu = self.quant_encoder_mu(q)
        z_q_sigma = self.quant_encoder_sigma(q)
        z_q = pyro.sample("z_q", dist.Normal(z_q_mu, z_q_sigma).to_event(2))
        
        c_q_dist = self.quant_encoder_class(q)
        c_q = pyro.sample("c_q", dist.OneHotCategoricalStraightThrough(logits=c_q_dist).to_event(1))




    
    def encode(self, x, q):
        batch_size = x.shape[0]
        
        x_flat = x.reshape(x.shape[0],-1)

        z_x_mu = self.rgbu_encoder_mu(x)
        z_x_sigma = self.rgbu_encoder_sigma(x)
        z_x = dist.Normal(z_x_mu, z_x_sigma).sample()
        
        c_x_dist = self.rgbu_encoder_class(x)
        c_x = dist.OneHotCategoricalStraightThrough(logits=c_x_dist).sample()


        z_q_mu = self.quant_encoder_mu(q)
        z_q_sigma = self.quant_encoder_sigma(q)
        z_q = dist.Normal(z_q_mu, z_q_sigma).sample()
        
        c_q_dist = self.quant_encoder_class(q)
        c_q = dist.OneHotCategoricalStraightThrough(logits=c_q_dist).sample()


        return  c_q, z_q, c_x, z_x
        
    def decode(self, c_q, z_q, c_x, z_x):
        batch_size = z_x.shape[0]
        
        combined_x_latents = torch.cat((c_x, z_x), -1)

        x_hat = self.rgbu_decoder(combined_x_latents)
        

        combined_q_latents = torch.cat((c_q, z_q), -1)
        
        q_hat = self.quant_decoder(combined_q_latents)
        q_err = torch.ones_like(q_hat,device=self.device)*(1.0/self.rgbu_samples)*self.q_err_gamma
        

        return x_hat, q_hat

    def reconstruct(self, x, q):
        c_q, z_q, c_x, z_x = self.encode(x,q)
        x_hat, q_hat = self.decode(c_q, z_q, c_x, z_x)
        return x_hat.detach(), q_hat.detach()





class JointVAE(nn.Module):
    def __init__(self, rgbu_dim, rgbu_samples, quant_dim, latent_dim, n_classes=10 ,x_err_gamma=1.0, q_err_gamma=1.0, hidden_dim=20, hidden_depth=3, dropout=0.0, skip_connection=True, device=None):
        super().__init__()
        self.rgbu_dim = rgbu_dim
        self.quant_dim = quant_dim
        self.latent_dim = latent_dim 
        self.n_classes = n_classes
        self.rgbu_samples = rgbu_samples
        self.rgbu_latent_dim = latent_dim
        self.quant_latent_dim = latent_dim*rgbu_samples
        self.patch_latent_dim = latent_dim*rgbu_samples
        self.device = device
        self.latent_eps=1e-3
        self.x_err_gamma = x_err_gamma
        self.q_err_gamma = q_err_gamma

        self.rgbu_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = rgbu_dim*rgbu_samples,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.quant_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = quant_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.joint_encoder_mu = MLP(
            input_dim       = quant_dim+rgbu_dim*rgbu_samples,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.joint_encoder_sigma = MLP(
            input_dim       = quant_dim+rgbu_dim*rgbu_samples,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = torch.sigmoid,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
        
        self.joint_encoder_class = MLP(
            input_dim       = quant_dim+rgbu_dim*rgbu_samples,
            output_dim      = n_classes,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
              
    def to(self,device):
        self.device = device
        return super().to(device)

    def model(self, x, q):
        pyro.module("rgbu_decoder", self.rgbu_decoder)
        pyro.module("quant_decoder", self.quant_decoder)

        batch_size = x.shape[0]
        
        prior_loc = torch.zeros((batch_size, self.latent_dim),device=self.device)
        prior_scale = torch.ones((batch_size, self.latent_dim),device=self.device)
        z = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(2))
        
        class_prior = torch.ones((batch_size,  self.n_classes),device=self.device)
        c = pyro.sample("c", dist.OneHotCategoricalStraightThrough(logits=class_prior).to_event(1))

        combined_latents = torch.cat((c, z), -1)

        x_hat = self.rgbu_decoder(combined_latents).reshape((batch_size,self.rgbu_samples,self.rgbu_dim))
        x_err = torch.ones_like(x,device=self.device)*self.x_err_gamma
        pyro.sample(
            'rgbu',
            dist.Normal(x_hat, x_err).to_event(3),
            obs=x
        ) 

        
        q_hat = self.quant_decoder(combined_latents)
        q_err = torch.ones_like(q_hat,device=self.device)*(1.0/self.rgbu_samples)*self.q_err_gamma
        pyro.sample(
            'quant',
            dist.Normal(q_hat,q_err).to_event(2),
            obs=q
        ) 




    def guide(self, x, q):
        pyro.module("joint_encoder_mu", self.joint_encoder_mu)
        pyro.module("joint_encoder_sigma", self.joint_encoder_sigma)
        pyro.module("joint_encoder_class", self.joint_encoder_class)

        batch_size = x.shape[0]
        
        x_flat = x.reshape(x.shape[0],-1)
        batch_data_concat = torch.cat((q, x_flat), 1)

        z_mu_flat = self.joint_encoder_mu(batch_data_concat)
        z_sigma_flat = self.joint_encoder_sigma(batch_data_concat)
        c_dist_flat = self.joint_encoder_class(batch_data_concat)

        z_mu = z_mu_flat.reshape(((batch_size, self.latent_dim)))
        z_sigma = z_sigma_flat.reshape(((batch_size, self.latent_dim)))+self.latent_eps
        c_dist = c_dist_flat.reshape(((batch_size,  self.n_classes)))

        z = pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(2))
        c = pyro.sample("c", dist.OneHotCategoricalStraightThrough(logits=c_dist).to_event(1))



    
    def encode(self, x, q):
        batch_size = x.shape[0]
        
        x_flat = x.reshape(x.shape[0],-1)
        batch_data_concat = torch.cat((q, x_flat), 1)

        z_mu_flat = self.joint_encoder_mu(batch_data_concat)
        z_sigma_flat = self.joint_encoder_sigma(batch_data_concat)
        c_dist_flat = self.joint_encoder_class(batch_data_concat)

        z_mu = z_mu_flat.reshape(((batch_size, self.latent_dim)))
        z_sigma = z_sigma_flat.reshape(((batch_size, self.latent_dim)))+self.latent_eps
        c_dist = c_dist_flat.reshape(((batch_size,  self.n_classes)))

        z = dist.Normal(z_mu, z_sigma).sample()
        c = dist.OneHotCategoricalStraightThrough(logits=c_dist).sample()



        return  c,z
        
    def decode(self, c, z):
        batch_size = z.shape[0]
        

        combined_latents = torch.cat((c, z), -1)

        x_hat = self.rgbu_decoder(combined_latents).reshape((batch_size,self.rgbu_samples,self.rgbu_dim))
        
        
        q_hat = self.quant_decoder(combined_latents)
        

        return x_hat, q_hat

    def reconstruct(self, x, q):
        c,z = self.encode(x,q)
        x_hat, q_hat = self.decode(c,z)
        return x_hat.detach(), q_hat.detach()




class NaiveFusionModel(nn.Module):
    def __init__(self, rgbu_dim, rgbu_samples, quant_dim, latent_dim, n_classes=10 ,x_err_gamma=1.0, q_err_gamma=1.0, hidden_dim=20, hidden_depth=3, dropout=0.0, skip_connection=True, device=None):
        super().__init__()
        self.rgbu_dim = rgbu_dim
        self.quant_dim = quant_dim
        self.latent_dim = latent_dim 
        self.n_classes = n_classes
        self.rgbu_samples = rgbu_samples
        self.rgbu_latent_dim = latent_dim
        self.quant_latent_dim = latent_dim*rgbu_samples
        self.patch_latent_dim = latent_dim*rgbu_samples
        self.device = device
        self.latent_eps=1e-3
        self.x_err_gamma = x_err_gamma
        self.q_err_gamma = q_err_gamma

        self.rgbu_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = rgbu_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.quant_decoder = MLP(
            input_dim       = latent_dim+n_classes,
            output_dim      = quant_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = F.relu,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.joint_encoder_mu = MLP(
            input_dim       = quant_dim+rgbu_dim,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

        self.joint_encoder_sigma = MLP(
            input_dim       = quant_dim+rgbu_dim,
            output_dim      = latent_dim,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = torch.sigmoid,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)
        
        self.joint_encoder_class = MLP(
            input_dim       = quant_dim+rgbu_dim,
            output_dim      = n_classes,
            hidden_dim      = hidden_dim,
            hidden_depth    = hidden_depth,
            activation      = F.softplus,
            output_activation = None,
            dropout         = dropout,
            skip_connection = skip_connection,
            device          = device)

              
    def to(self,device):
        self.device = device
        return super().to(device)

    def model(self, x, q):
        pyro.module("rgbu_decoder", self.rgbu_decoder)
        pyro.module("quant_decoder", self.quant_decoder)


        batch_size = x.shape[0]
        
        prior_loc = torch.zeros((batch_size, self.rgbu_samples, self.latent_dim),device=self.device)
        prior_scale = torch.ones((batch_size, self.rgbu_samples, self.latent_dim),device=self.device)
        z = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(3))
        
        alpha = torch.ones((batch_size, self.n_classes),device=self.device)
        class_dist = pyro.sample("c_dist", dist.Dirichlet(alpha).to_event(1))

        c_prior =  class_dist.repeat(self.rgbu_samples,1,1).swapaxes(0,1)
        c = pyro.sample("c", dist.OneHotCategoricalStraightThrough(logits=c_prior).to_event(2))
        

        z_nested_batch_agg = z.reshape((batch_size*self.rgbu_samples, self.latent_dim))
        c_nested_batch_agg = c.reshape((batch_size*self.rgbu_samples, self.n_classes))
        nested_batch_combined_latents = torch.cat((c_nested_batch_agg, z_nested_batch_agg), 1)

        x_hat_flat = self.rgbu_decoder(nested_batch_combined_latents)
        x_hat = x_hat_flat.reshape(x.shape)
        x_err = torch.ones_like(x,device=self.device)*self.x_err_gamma
        pyro.sample(
            'rgbu',
            dist.Normal(x_hat, x_err).to_event(3),
            obs=x
        ) 

        q_hat_flat = self.quant_decoder(nested_batch_combined_latents)
        q_hat = q_hat_flat.reshape((batch_size,self.rgbu_samples,self.quant_dim)).mean(1)
        q_err = torch.ones_like(q_hat,device=self.device)*(1.0/self.rgbu_samples)*self.q_err_gamma
        pyro.sample(
            'quant',
            dist.Normal(q_hat,q_err).to_event(2),
            obs=q
        ) 


    def guide(self, x, q):
        pyro.module("joint_encoder_mu", self.joint_encoder_mu)
        pyro.module("joint_encoder_sigma", self.joint_encoder_sigma)
        pyro.module("joint_encoder_class", self.joint_encoder_class)

        batch_size = x.shape[0]
        
        x_batch_agg = x.reshape((batch_size*self.rgbu_samples, self.rgbu_dim))
        q_duplicated_batch_agg = q.repeat(1,self.rgbu_samples).reshape((batch_size*self.rgbu_samples, self.quant_dim))
        batch_data_concat = torch.cat((q_duplicated_batch_agg, x_batch_agg), 1)

        z_mu_flat = self.joint_encoder_mu(batch_data_concat)
        z_sigma_flat = self.joint_encoder_sigma(batch_data_concat)
        c_dist_flat = self.joint_encoder_class(batch_data_concat)

        z_mu = z_mu_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))
        z_sigma = z_sigma_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))+self.latent_eps
        c_estimate = c_dist_flat.reshape(((batch_size, self.rgbu_samples, self.n_classes)))

        z = pyro.sample("z", dist.Normal(z_mu, z_sigma).to_event(3))
        c = pyro.sample("c", dist.OneHotCategoricalStraightThrough(logits=c_estimate).to_event(2))
        class_dist = pyro.sample("c_dist", dist.Dirichlet(c.mean(1)+self.latent_eps).to_event(1))


    def encode(self, x, q):
        batch_size = x.shape[0]
        
        x_batch_agg = x.reshape((batch_size*self.rgbu_samples, self.rgbu_dim))
        q_duplicated_batch_agg = q.repeat(1,self.rgbu_samples).reshape((batch_size*self.rgbu_samples, self.quant_dim))
        batch_data_concat = torch.cat((q_duplicated_batch_agg, x_batch_agg), 1)

        z_mu_flat = self.joint_encoder_mu(batch_data_concat)
        z_sigma_flat = self.joint_encoder_sigma(batch_data_concat)
        c_dist_flat = self.joint_encoder_class(batch_data_concat)

        z_mu = z_mu_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))
        z_sigma = z_sigma_flat.reshape(((batch_size, self.rgbu_samples, self.latent_dim)))+self.latent_eps
        c_estimate = c_dist_flat.reshape(((batch_size, self.rgbu_samples, self.n_classes)))

        z = dist.Normal(z_mu, z_sigma).sample()
        c = dist.OneHotCategoricalStraightThrough(logits=c_estimate).sample()

        return  c, z
        
    def decode(self, c, z):
        batch_size = z.shape[0]
        
        combined_latents = torch.cat((c, z), -1)

        z_nested_batch_agg = z.reshape((batch_size*self.rgbu_samples, self.latent_dim))
        c_nested_batch_agg = c.reshape((batch_size*self.rgbu_samples, self.n_classes))
        nested_batch_combined_latents = torch.cat((c_nested_batch_agg, z_nested_batch_agg), 1)

        x_hat_flat = self.rgbu_decoder(nested_batch_combined_latents)
        x_hat = x_hat_flat.reshape((batch_size,self.rgbu_samples, self.rgbu_dim))

        q_hat_flat = self.quant_decoder(nested_batch_combined_latents)
        q_hat = q_hat_flat.reshape((batch_size,self.rgbu_samples,self.quant_dim)).mean(1)

        return x_hat, q_hat

    def reconstruct(self, x, q):
        c, z = self.encode(x,q)
        x_hat, q_hat = self.decode(c, z)
        return x_hat.detach(), q_hat.detach()