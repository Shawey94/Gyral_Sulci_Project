#group analysis sulci gyri ratio
import torch
import numpy as np
from scipy.stats import zscore
import h5py
import matplotlib.pyplot as plt
import time
import os
import gc

def ReadLabel(sub_id):
    vtk_filename = "/media/shawey/SSD8T/GyraSulci_Motor/H5_vtk/{}.vtk".format(sub_id)
    flag = 0
    label = []
    for line in open(vtk_filename):
          if "LOOKUP_TABLE gyri_sulci_roi" in line:
              flag =  1
              continue
          if flag == 0:
             continue
          label.append(int(float(line.strip())))
    return np.array(label)

def ConstructGraph(gyri_network, sulci_network,unknown_network,gyri_signal,sulci_signal, gyri_pos, sulci_pos, roi, mask_label):
    matrix = np.zeros((35559,35559), dtype='i1' )  #first 17232 gyri, 18327 sulci
    gyri_sulci_network = np.concatenate((gyri_network, sulci_network))
    gyri_sulci_indices = np.array(np.where(gyri_sulci_network == 1.0)).reshape(1,-1)   # indices <= 35559
    
    #--------------------------------------------------------------------------------
    all_signal = np.zeros(64984) 
    roi_signal = np.zeros(59412)
    #roi_signal[mask_label == 1] = 0
    #roi_signal[mask_label == -1] = 0
    #all_signal[roi[:] == True] = roi_signal
    
    for pos in gyri_sulci_indices[0,:]:            
        if(pos < 17232):   #17232 gyri
            roi_signal[gyri_pos[0,pos]] = 1 #abs(gyri_signal[pos])
        elif(pos < 35559):
            #print(sulci_pos[0,pos-17232])
            roi_signal[sulci_pos[0,pos-17232]] = -1  #abs(sulci_signal[pos-17232])        
    all_signal[roi[:] == True] = roi_signal        
    #---------------------------------------------------------------------------------

    for i in gyri_sulci_indices[0,:]:
        for j in gyri_sulci_indices[0,:]:
            matrix[i,j] = 1

    return matrix, all_signal

def LoadParams():
    filename = "/media/shawey/SSD8T/GyraSulci_Motor/H5_vtk/100408.h5"
    f1 = h5py.File(filename,'r+')
    roi = f1['roi']
    label = ReadLabel('100408')
    mask_label = label[roi[:]==True]
    return roi, mask_label

def main(path, pkl_name, id_part, comp_id,THRESHOLD):
    w1 = zscore(torch.load(path+pkl_name,torch.device("cpu")))[:,:59412]
    roi, mask_label = LoadParams()
    matrix = np.zeros((35559,35559),dtype='i1')
    signal = np.zeros(64984)

    tick1 = time.time()
    gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
    sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 
    sub_gyri_num = 0
    sub_sulci_num = 0
    gyri_sulci_num = 0
    for i in comp_id:
        print(i)
        gyri_network = w1[i][mask_label==1]                  #gyri 17232
        sulci_network = w1[i][mask_label==-1]    #sulci 18327
        THRESHOLD = THRESHOLD * np.concatenate((gyri_network, sulci_network)).max()

        # this is in mask_label level (including gyri, sulci and unknown),not in roi level
        gyri_network = np.where(gyri_network >= THRESHOLD,  gyri_network, 0)
        gyri_network = np.where(gyri_network < THRESHOLD, gyri_network, 1 )  
        gyri_filtered_num = np.array((np.where(gyri_network == 1.0))).shape[1]
        print('gyri filtered num ', gyri_filtered_num)

        unknown_network = w1[i][mask_label==0]                  
        unknown_pos = np.array(np.where(mask_label == 0)).reshape(1,-1) 
        # this is in mask_label level (including gyri, sulci and unknown),not in roi level
        unknown_network = np.where(unknown_network >= 10000,  unknown_network, 0)

        sulci_network = np.where(sulci_network >= THRESHOLD,  sulci_network, 0)
        sulci_network = np.where(sulci_network < THRESHOLD, sulci_network, 1 ) 
        sulci_filtered_num = np.array((np.where(sulci_network == 1.0))).shape[1]
        print('sulci filtered num ',sulci_filtered_num)

        gyri_sulci_network = np.concatenate((gyri_network, sulci_network))
        gyri_sulci_filtered_num = np.array((np.where(gyri_sulci_network == 1.0))).shape[1]

        sub_gyri_num += gyri_filtered_num
        sub_sulci_num += sulci_filtered_num
        gyri_sulci_num += gyri_sulci_filtered_num + 0.000001

    tick2 = time.time()
    print('time usage ', tick2-tick1)
    print("matrix max",np.max(matrix))
    #plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'.jpeg',\
    #        matrix, vmax=5, vmin=-5, cmap ='coolwarm')
    if( sub_gyri_num < 100 or sub_sulci_num < 100):
        return 0, 0
    return sub_gyri_num / gyri_sulci_num, sub_sulci_num / gyri_sulci_num


#USE gyri_weight generated from haxing
path = "/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/figs/"
pkls = os.listdir(path)
THRESHOLDs = [0.8]

tr_type = 'comm_spa'
if tr_type == "full":
    tr_index = range(100)
elif tr_type == "comm_spa":
    tr_index = range(85, 100)
elif tr_type == "comm_tem":
    tr_index = range(0,10)
elif tr_type == "indv":
    tr_index = range(10,85)

roi, mask_label = LoadParams()
gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 
gyri_ratio_comm_spa = []
sulci_ratio_comm_spa = []
for THRESHOLD in THRESHOLDs:
    print('threshold is ', THRESHOLD)
    for comp_id in tr_index:
        gyri_ratio = 0 
        sulci_ratio = 0
        cnt = 0
        for pkl in pkls:
            print(pkl)
            sub_id = pkl.split('_')[0]
            part = pkl.split('_')[1]
            id_part = sub_id + '_' + part
            temp_gyri_ratio, temp_sulci_ratio = main(path,pkl,id_part, [comp_id], THRESHOLD)
            if(temp_sulci_ratio == 0 or temp_gyri_ratio == 0):
                continue
            gyri_ratio += temp_gyri_ratio
            sulci_ratio += temp_sulci_ratio
            cnt += 1
        
        gyri_ratio_comm_spa.append( gyri_ratio/cnt )
        sulci_ratio_comm_spa.append( sulci_ratio/cnt )
  
print('gyri_ratio_' +tr_type+ ' {}'.format(gyri_ratio_comm_spa) )
print('sulci_ratio_' +tr_type+ ' {}'.format(sulci_ratio_comm_spa) )
print('gyri_ratio_' +tr_type+ ' {} {}'.format(np.array(gyri_ratio_comm_spa).mean(), np.array(gyri_ratio_comm_spa).std()) )
print('sulci_ratio_'+tr_type+' {} {}'.format(np.array(sulci_ratio_comm_spa).mean(),np.array(sulci_ratio_comm_spa).std() ) )
