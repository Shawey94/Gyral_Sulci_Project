from __future__ import print_function
import torch
import numpy as np
from parameters import *
import nibabel as nib
import h5py
import scipy.io as scio
from torch import nn

def temporal_corr_score(x, y, num): #x and y is 3d data, be careful using torch.mean()
    comm_temp_random_indices = torch.from_numpy(np.random.rand(num)).to(device).reshape((1, num ))
    print(x)
    print(y)
    mean_x = torch.mean(x, dim = 1, keepdim= True) 
    mean_y = torch.mean(y, dim = 1, keepdim= True) 
    vx = x - mean_x
    vy = y - mean_y
    print(vx * vy)
    cost = 1 - (torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2,dim=1)) * torch.sqrt(torch.sum(vy ** 2,dim=1))))
    print(cost)
    cost = comm_temp_random_indices - (torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2,dim=1)) * torch.sqrt(torch.sum(vy ** 2,dim=1)))) * comm_temp_random_indices
    print(cost)
    return torch.mean(cost)


'''
def temporal_orthogonal_regularization(matrix1, matrix2, beta=2/opt.num_comm/(opt.num_comm-1)   ): #matrxi size 284 * 30 (30 common temporal)
    # beta = 1/ opt.num_comm                                                      
    # beta * (||W^T.W * (1-I)||_F)^2 or 
    # beta * (||W.W.T * (1-I)||_F)^2
    # 若 H < W,可以使用前者， 若 H > W, 可以使用后者，这样可以适当减少内存
            
    N, H, W = matrix1.shape  # 1, 284, 30
            
    weight_squared = torch.bmm( matrix1.permute(0, 2, 1),matrix2).to(device) # (N ) * W * W
    
    ones = torch.ones(N, W, W).to(device)
    diag = torch.eye(W, dtype=torch.float32).to(device) # (N) * W * W
            
    loss_orth = ((weight_squared * (ones-diag)) ** 1).sum()
            
    return loss_orth * beta


test_input1 = torch.rand(1,3,2).to(device)
test_input2 = torch.rand(1,3,2).to(device)

corr = temporal_corr_score(test_input1[:,:,0:4],test_input2[:,:,0:4], 2)
print(corr)
#ot = temporal_orthogonal_regularization(test_input1[:,:,0:4],test_input2[:,:,0:4])
#print(ot)

_,_,num_col = test_input1.shape
temporal_norm = torch.sum(torch.sum(test_input1 * test_input1, dim=1))
print(temporal_norm/ num_col) 
temp = 0
for i in range(num_col):
    temp += (test_input1[:,:,i] * test_input1[:,:,i]).sum()
    print(temp/num_col) 


test_input1 = np.array(test_input1)
test_input2 = np.array(test_input2)
temp = abs(np.corrcoef(test_input1[:,:,0],test_input2[:,:,0])[0,1]) + abs(np.corrcoef(test_input1[:,:,1],test_input2[:,:,1])[0,1])+\
          abs(np.corrcoef(test_input1[:,:,2],test_input2[:,:,2])[0,1]) +abs(np.corrcoef(test_input1[:,:,3],test_input2[:,:,3])[0,1])

print(temp/4)
'''

def check_signal_similarity():
    filtered_case = []
    patch_len = 288   # 284 * 28800 divide into 284 * 288, (28800 / 288 = 100) patches
    #datapath='/media/shawey/SSD8T/HCP/HCP3T4mm/HCP3T4mm_fMRI/HCP_1200_tfMRI_4mm'
    subs = os.listdir(opt.datafolder)
    max_mask_len = 28800
    label_file = '/media/shawey/SSD8T/HCP/HCP3T4mm/data_code_bm_template_task_label/MOTOR_label.mat'
    data = scio.loadmat(label_file)
    #print(data)
    label = data['Label']
    time_point, num_type = label.shape   
    for sub in subs:
        sub_folder = opt.datafolder + '/' + sub
       
        with h5py.File(sub_folder, 'r') as f:
            #print(f.keys())
            fmri = f.get('MOTOR')
            fmri_data = np.array(fmri)[0:opt.time_length,:]
        if(max_mask_len > fmri_data.shape[1]):
            add_array = np.zeros((opt.time_length, max_mask_len - fmri_data.shape[1]))
            fmri_data1 = np.concatenate((fmri_data, add_array), axis=1)
        else:
            fmri_data1 = fmri_data[:,0:max_mask_len]

        print('fmri_data1.shape is', fmri_data1.shape)


        for j in range(100): #100 patches
            cnt = 0
            temp = fmri_data1[:,j*patch_len:(j+1)* patch_len]
            for k in range(patch_len):
                signal = temp[:,k]
                for i in range(num_type):
                    if( abs((np.corrcoef(label[:, i], signal)[0,1])) > 0.5 ):
                        cnt += 1
            print('cnt is',cnt)
            if (cnt > 100):
                print('sub {}, patch {}'.format(sub, j))
                filtered_case.append(sub)
                filtered_case.append(j)
    return filtered_case

def find_mask_index(fmri_data, task_design):
    task_design = task_design.reshape(opt.time_unit,-1)
    print('task_design shape is {}'.format(task_design.shape))
    time, voxel_num = fmri_data.shape
    mask_index = []
    for i in range(voxel_num):
        signal = fmri_data[:,i]
        for j in range( task_design.shape[1]):
            if( abs(np.corrcoef(signal, task_design[:,j])[0,1]) > 0.4 ):
                mask_index.append(i)
    return mask_index

'''
comm_spa_loss_crietion = nn.L1Loss()
def comm_spa_loss(comm_spa1, comm_spa2, num):   #comm_spa size: 1 opt.spa_comm 28600
    comm_spa_random_indices = torch.from_numpy(np.random.rand(num)).to(device).reshape((1,num, 1 ))
    print(comm_spa_random_indices)
    print(comm_spa1)
    comm_spa1 = comm_spa1 * comm_spa_random_indices
    print(comm_spa1)
    comm_spa2 = comm_spa2 * comm_spa_random_indices
    spatial_loss =  comm_spa_loss_crietion(comm_spa1,\
                comm_spa2)
    return spatial_loss
'''

def comm_spa_loss(comm_spa1, comm_spa2):   #comm_spa size: 1 opt.spa_comm 28600
    mean_x = torch.mean(comm_spa1, dim = 2, keepdim= True) 
    mean_y = torch.mean(comm_spa2, dim = 2, keepdim= True) 
    print(comm_spa1)
    print(comm_spa2)
    print(mean_x)
    print(mean_y)
    vx = comm_spa1 - mean_x
    vy = comm_spa1 - mean_y
    print(vx)
    print(vy)
    print(vx * vy)
    print(torch.sum(vx * vy, dim=2))
    print(torch.sum(vx ** 2,dim=2))
    cost = 1 - (torch.sum(vx * vy, dim=2) / (torch.sqrt(torch.sum(vx ** 2,dim=2)) * torch.sqrt(torch.sum(vy ** 2, dim=2)))) 
    return torch.mean(cost)


if __name__== '__main__':
    #filtered_case = check_signal_similarity()
    #print(filtered_case)

    num = 5
    comm_spa1 = torch.rand(1, 1, 284).to(device)
    comm_spa2 = torch.rand(1, 1, 284).to(device)
    loss = comm_spa_loss(comm_spa1, comm_spa2)
    print(loss)
