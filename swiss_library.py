import os
from pandas import read_csv
from numpy import dstack
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from warnings import warn
from torch import nn
import math
from torch.utils.tensorboard import SummaryWriter # type: ignore

def dataset_loader(x_data, y_data, subject, mode):
    
    volunteers = []
    x, y, sub = torch.empty(0), torch.empty(0), torch.empty(0)
    # get the unique subjects as integers
    subject_list = np.unique(subject).astype(int)
    # print(f'length subject_list: {len(subject_list)}')

    # sort the subjects
    subject_list = np.sort(subject_list)
    total_data = 0

    for crt_path in subject_list:
        # print(f'Subject: {crt_path}')
        volunteer = crt_path
        # get indexes of the current subject
        idx = np.where(subject == crt_path)
        # remove the first dimension of the indexes
        idx = idx[0]
        len_data = len(idx)
        # print(f'Number of observations: {len_data}')
        total_data += len_data
        # get the data and labels of the current subject
        crt_x, crt_y = x_data[idx], y_data[idx]
        crt_sub = subject[idx]

        crt_x = torch.FloatTensor(crt_x)
        crt_y = torch.Tensor(crt_y)
        crt_sub = torch.Tensor(crt_sub)

        x = torch.cat([x, crt_x])
        y = torch.cat([y, crt_y])
        sub = torch.cat([sub, crt_sub])


        volunteers.append(volunteer)

    print(f'[{mode} Loader] {mode} observations : {x.shape[0]}')
    print(f'[{mode} Loader] {mode} labels : {y.shape[0]}')
    print(f'[{mode} Loader] {mode} volunteers : {len(volunteers)}')

    data_dict = {'x':x, 'y':y, 'sub':sub}
    obs = x.shape[0]

    return data_dict, obs, volunteers

def custom_normalize(train_x):
    """z-normalization

    Args:
        train_x (torch.FloatTensor): [shape : N, S, T]
    """
    N,S,T = train_x.shape
    normalized_data = torch.zeros((N,S,T))
    
    means, stds = [], []
    for i in range(S):
        crt_signal = train_x[:,i,:]
        mean = crt_signal.mean()# mean of signal i
        std = crt_signal.std()# std of signal i
        means.append(mean)
        stds.append(std)
        normalized_data[:,i,:] = (crt_signal - mean) / std
    
    return normalized_data, means, stds

def custom_norm_transform(data, means, stds):
    N,S,T = data.shape
    normalized_data = torch.zeros((N,S,T))
    
    for i in range(S):
        crt_signal = data[:,i,:]
        mean = means[i]
        std = stds[i]
        normalized_data[:,i,:] = (crt_signal-mean) / std
    return normalized_data

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length `L`, consisting of subsequences of average length lm, masking with 0s a `masking_ratio`
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.zeros(L, dtype=bool)

    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    if L == 33:
        p_u = (p_m * masking_ratio / (1 - masking_ratio))  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    else:
        p_u = (p_m * masking_ratio / (1 - masking_ratio))
        
    p = [p_u, p_m]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() < masking_ratio)  # state 0 means not masking, 1 means masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return torch.tensor(keep_mask)

class SWISS_dataset(Dataset):
    def __init__(self, data_dict, mask_type, masking_rate):
        """
        x shape : (# of obs, # of signals, length)
        y shape : (# of obs,)
        sub shape : (# of obs,)
        """
        self.x, self.sub = data_dict['x'], data_dict['sub']
        self.aug_list = ['jitter', 'scaling', 'rotation', 'permutation', 'channel_shuffle']
        self.mask_type = mask_type
        self.masking_rate = masking_rate

    def __getitem__(self, index):        

        input = self.x[index]
        subj = self.sub[index]
            
        mask = self.create_mask(input.shape, self.mask_type, self.masking_rate)

        x_1, x_2 = input, input
            
        return x_1, x_2, mask, subj

    def __len__(self):
        return self.x.shape[0]
    
    def create_mask(self, mask_shape, mask_type, p):
        """
        state 0 means not masking, 1 means masking """
        s, l = mask_shape # s: number of signals, l: length of signal
        mask_l = int(np.ceil(l*p)) # length of mask (ex: 128*0.1 = 12.8 -> 13)
        if mask_type == 'random':
            mask = torch.rand(mask_shape).ge(1-p)
        
        elif mask_type == 'geom':
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            for i in range(s):
                mask[i, :] = geom_noise_mask_single(l, lm=mask_l, masking_ratio=p)
                
        elif mask_type == 'poisson':
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            for i in range(s):
                try:
                    crt_l = np.random.poisson(mask_l)
                    crt_point = np.random.randint(0,l-crt_l+1)
                    mask[i,crt_point:crt_point+crt_l] = 1
                except:
                    mask[i,:] = 1
                
        elif mask_type == 'same':
            mask = torch.zeros(mask_shape, dtype=torch.bool)
            for i in range(s):
                crt_point = np.random.randint(0,l-mask_l+1)
                mask[i,crt_point:crt_point+mask_l] = 1
        
        else:
            raise Exception(f'mask type error: there is no {mask_type} mask type')
        
        return mask

class Downstream_dataset(Dataset):
    def __init__(self, data_dict):
        """
        x shape : (# of obs, # of signals, length)
        y shape : (# of obs,)
        """
        self.x, self.y = data_dict['x'], data_dict['y']

    def __getitem__(self, index):        

        input = self.x[index]
        target = self.y[index]
            
        return input, target

    def __len__(self):
        return self.x.shape[0]
    
class Saver:
    def __init__(self, path):
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

            warn(f'{path} does not exist. Creating.')

    def checkpoint(self, tag, payload, is_best=False):
        checkpoint_path = self.get_path(tag, is_best)

        with open(checkpoint_path, "wb+") as fp:
            _payload = payload.state_dict()
            torch.save(_payload, fp)

    def get_path(self, tag, is_best=False):
        if is_best:
            fname = f'{tag}.best'
        else:
            fname = f'{tag}.pt'
        checkpoint_path = os.path.join(self.path, fname)

        return checkpoint_path

    def load(self, tag, model, is_best=False):
        checkpoint_path = self.get_path(tag, is_best)
        
        if os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            _payload = model.state_dict()
            _payload.update(payload)

            model.load_state_dict(_payload)
            print('All keys matched successfully...')

        else:
            warn(f'Error: {checkpoint_path} No Weights loaded')

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))

        return writer
    
def initialize_weights(model: nn.Module, activation='relu'):
    """Initialize trainable weights."""

    for _, m in model.named_modules():
        
        if activation == 'relu':
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity=activation)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                try:
                    nn.init.constant_(m.bias, 1)
                except AttributeError:
                    pass
        
        else:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                try:
                    nn.init.constant_(m.bias, 1)
                except AttributeError:
                    pass