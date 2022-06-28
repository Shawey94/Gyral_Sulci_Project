import itertools
import random
import torch
import numpy as np
import os
import nibabel as nib
from parameters import *
import matplotlib.pyplot as plt
import scipy.io as scio
from torch import nn

time_len = opt.time_length
time_unit = opt.time_unit
def make_data(fmri_data1, fmri_data2):
    data = list(zip(fmri_data1, fmri_data2))

    return data

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def temporal_corr_score(x, y): #x and y is 3d data, be careful using torch.mean()
    mean_x = torch.mean(x, dim = 1, keepdim= True) 
    mean_y = torch.mean(y, dim = 1, keepdim= True) 
    vx = x - mean_x
    vy = y - mean_y
    cost = 1 - (torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2,dim=1)) * torch.sqrt(torch.sum(vy ** 2,dim=1)))) 
    return torch.sum(cost)


def GenePair(subs):  
    ''' 
    test_num = int(len(subs) * opt.test_ratio)
    train_num = len(subs) - test_num
    tra_PairList = list(itertools.combinations(subs[0:train_num],2))
    test_subs = subs[train_num:]
    '''
    tra_PairList = list(itertools.combinations(subs,2))
    return tra_PairList

def temporal_basis_norm(matrix, beta=1): #matrxi size 284 * 100
    
    # beta * (||W^T.W * (I)||_F)^2 or 
    # beta * (||W.W.T * (I)||_F)^2
    # 若 H < W,可以使用前者， 若 H > W, 可以使用后者，这样可以适当减少内存
            
    N, H, W = matrix.shape #e.g matrix shape is 1, 284, 100    
    weight_squared = torch.bmm( matrix.permute(0, 2, 1),matrix) # (N ) * W * W   
    zeros = torch.zeros(N , W, W, dtype=torch.float32).to(device)  # (N) * W * W
    diag = torch.eye(W, dtype=torch.float32).to(device) # (N) * W * W
    loss_orth = ((weight_squared * (zeros + diag)) ** 1).sum()
            
    return loss_orth * beta/W

#mask_path='/home/xiaow/IJCAI2022/HCP3T4mm'
def find_max_mask_length():
    subs = os.listdir(opt.mask_path)
    max_mask_len = 0
    for sub in subs:
        if (sub =='MNI152_T1_4mm_brain_mask.nii.gz'):
            mask = nib.load(opt.mask_path +'/'+ sub )
            mask_data = mask.get_fdata()[:,:,:]
            indices = np.where(mask_data != 0.0)
            temp = mask_data[indices]
            if (len(temp) > max_mask_len):
                max_mask_len = len(temp)
    return max_mask_len

'''
comm_spa_loss_crietion = nn.L1Loss()
def comm_spa_loss(comm_spa1, comm_spa2, num):   #comm_spa size: 1 opt.spa_comm 28600
    comm_spa_random_indices = torch.from_numpy(np.random.rand(num)).to(device).reshape((1, num, 1 ))
    comm_spatial_loss =  comm_spa_loss_crietion(comm_spa1 * comm_spa_random_indices,\
                comm_spa2 * comm_spa_random_indices)
    return comm_spatial_loss
'''

def comm_spa_loss(comm_spa1, comm_spa2):   #comm_spa size: 1 opt.spa_comm 28600
    mean_x = torch.mean(comm_spa1, dim = 2, keepdim= True) 
    mean_y = torch.mean(comm_spa2, dim = 2, keepdim= True) 
    vx = comm_spa1 - mean_x
    vy = comm_spa1 - mean_y
    cost = 1 - (torch.sum(vx * vy, dim=2) / (torch.sqrt(torch.sum(vx ** 2,dim=2)) * torch.sqrt(torch.sum(vy ** 2, dim=2)))) 
    return torch.sum(cost)



        



