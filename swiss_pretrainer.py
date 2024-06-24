from calendar import c
import os
import json
import time
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from zmq import device
from swiss_library import Saver, TensorboardSummary
from signal_token_transformer import STT
from swiss_classifier import Projector, Predictor, ReconstructionHead


class SWISS_trainer(object):
    def __init__(self, pretrain_loader, save_path: str, pretrain_dict):

        # define save directory
        self.save_path = os.path.join(save_path, f'/pretrain')
        self.check_path = os.path.join(self.save_path, 'pretrain_ckpt')
        self.summary_path = os.path.join(self.save_path, 'pretrain_runs')

        # define dataloader
        self.pretrain_loader = pretrain_loader

        # Define saver
        self.saver = Saver(self.check_path)

        # define Tensorboard summary
        self.summary = TensorboardSummary(self.summary_path)
        self.writer = self.summary.create_summary()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define Transformer online network and target network
        self.online_encoder = STT(gru_hid_dim=pretrain_dict['gru_hid_dim'], 
                                  gru_input_size=pretrain_dict['gru_input_size'], 
                                  gru_layers=pretrain_dict['gru_layers'], 
                                  gru_dropout=pretrain_dict['gru_dropout'], 
                                  bidirectional=pretrain_dict['bidirectional'], 
                                  num_signals=pretrain_dict['num_signals'], emb_dim=pretrain_dict['emb_dim'], 
                                  device=self.device, 
                                  emb_dropout=pretrain_dict['emb_dropout'], 
                                  depth=pretrain_dict['depth'], heads=pretrain_dict['heads'], 
                                  head_dim=pretrain_dict['head_dim'], 
                                  transformer_mlp_dim=pretrain_dict['transformer_mlp_dim'], 
                                  dropout=pretrain_dict['dropout'], 
                                  signal_emb=pretrain_dict['signal_emb']).to(self.device)
        # self.online_projector = Projector(args).to(self.device)
        # self.online_predictor = Predictor(args).to(self.device)
        self.online_projector = Projector(emb_dim=pretrain_dict['emb_dim'], 
                                          num_signals=pretrain_dict['num_signals'], 
                                          proj_hiddim=pretrain_dict['proj_hiddim'], 
                                          proj_dim=pretrain_dict['proj_dim']).to(self.device)
        self.online_predictor = Predictor(proj_hiddim=pretrain_dict['proj_hiddim'], 
                                          proj_dim=pretrain_dict['proj_dim']).to(self.device)
        
        self.target_encoder = STT(gru_hid_dim=pretrain_dict['gru_hid_dim'], 
                                  gru_input_size=pretrain_dict['gru_input_size'], 
                                  gru_layers=pretrain_dict['gru_layers'], 
                                  gru_dropout=pretrain_dict['gru_dropout'], 
                                  bidirectional=pretrain_dict['bidirectional'], 
                                  num_signals=pretrain_dict['num_signals'], emb_dim=pretrain_dict['emb_dim'], 
                                  device=self.device, 
                                  emb_dropout=pretrain_dict['emb_dropout'], 
                                  depth=pretrain_dict['depth'], heads=pretrain_dict['heads'], 
                                  head_dim=pretrain_dict['head_dim'], 
                                  transformer_mlp_dim=pretrain_dict['transformer_mlp_dim'], 
                                  dropout=pretrain_dict['dropout'], 
                                  signal_emb=pretrain_dict['signal_emb']).to(self.device)
        self.target_projector = Projector(emb_dim=pretrain_dict['emb_dim'], 
                                          num_signals=pretrain_dict['num_signals'], 
                                          proj_hiddim=pretrain_dict['proj_hiddim'], 
                                          proj_dim=pretrain_dict['proj_dim']).to(self.device)
        self.set_required_grid(self.target_encoder, False)
        self.set_required_grid(self.target_projector, False)

        self.recon_head = ReconstructionHead(num_signals=pretrain_dict['num_signals'], 
                                             input_dim=pretrain_dict['input_dim'], 
                                             feature_dim=pretrain_dict['feature_dim']).to(self.device)
        
        self.m = pretrain_dict['m']
        self.criterion = nn.MSELoss().to(self.device)

        


        
    def set_required_grid(self, model, val):
        for p in model.parameters():
            p.requires_grad = val