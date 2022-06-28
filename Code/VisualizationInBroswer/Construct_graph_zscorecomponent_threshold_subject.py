#generate matrix and map2vtk
import torch
import numpy as np
from scipy.stats import zscore
import h5py
import matplotlib.pyplot as plt
import time
import os
from vtk_utils import *
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
            roi_signal[gyri_pos[0,pos]] += 1 #abs(gyri_signal[pos])
        elif(pos < 35559):
            #print(sulci_pos[0,pos-17232])
            roi_signal[sulci_pos[0,pos-17232]] += -1  #abs(sulci_signal[pos-17232])        
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

def main(path, pkl_name, save_path, id_part, comp_id,THRESHOLD):
    temp = torch.load(path+pkl_name,torch.device("cpu")).t()
    w1 = zscore(temp)[:59412,:]
    w1 = np.transpose(w1)
    #w1 = zscore(torch.load(path+pkl_name,torch.device("cpu")))[:,:59412]
    roi, mask_label = LoadParams()

    tick1 = time.time()
    gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
    sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 

    i = comp_id
    print(i)

    gyri_network = w1[i][mask_label==1]      #gyri 17***
    sulci_network = w1[i][mask_label==-1]    #sulci 18***
    THRESHOLD = THRESHOLD 
    print('THRESHOLD ',THRESHOLD)

    gyri_signal = gyri_network
    # this is in mask_label level (including gyri, sulci and unknown),not in roi level
    gyri_network = np.where(gyri_network >= THRESHOLD,  gyri_network, 0)
    gyri_network = np.where(gyri_network < THRESHOLD, gyri_network, 1 )  
    print('gyri filtered num ',np.array(np.where(gyri_network == 1.0)).shape )

    unknown_network = w1[i][mask_label==0]                 
    unknown_pos = np.array(np.where(mask_label == 0)).reshape(1,-1) 
    # this is in mask_label level (including gyri, sulci and unknown),not in roi level
    unknown_network = np.where(unknown_network >= 10000,  unknown_network, 0)

    sulci_signal = sulci_network
    sulci_network = np.where(sulci_network >= THRESHOLD,  sulci_network, 0)
    sulci_network = np.where(sulci_network < THRESHOLD, sulci_network, 1 )  
    print('sulci filtered num ',np.array(np.where(sulci_network == 1.0)).shape )

    temp_matrix, temp_signal= ConstructGraph(gyri_network, sulci_network,unknown_network,gyri_signal,sulci_signal, gyri_pos, sulci_pos,roi, mask_label)
  
    tick2 = time.time()
    print('time usage ', tick2-tick1)
    
    return temp_matrix, temp_signal
    
#USE gyri_weight generated from haxing
path = "/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/figs/"
pkls = os.listdir(path)
save_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/group_wise'
THRESHOLDs = [0.3]

tr_type = 'comm_spa'
if tr_type == "full":
    tr_index = range(100)
elif tr_type == "comm_spa":
    tr_index = range(0, 15)
elif tr_type == "comm_tem":
    tr_index = range(15,30)
elif tr_type == "indv":
    tr_index = range(30,100)

reco_subs = []

for THRESHOLD in THRESHOLDs:
    print('threshold is ', THRESHOLD)
    scalars = []
    labels = []
    for pkl in pkls:  
        if (not os.path.isfile(path+pkl)):
            continue
        matrix = np.zeros((35559,35559),dtype='i1')
        signal = np.zeros(64984)
        print(pkl)
        sub_id = pkl.split('_')[0]
        part = pkl.split('_')[1]
        id_part = sub_id + '_' + part
        #if( sub_id in reco_subs):
        #    continue
        reco_subs.append(sub_id)
        for comp_id in (tr_index):
            subject_matrix, subject_signal = main(path,pkl,save_path,id_part,comp_id, THRESHOLD)
            matrix += subject_matrix
            signal += subject_signal
        vmin = -20
        vmax = 20
        matrix[0:17232,0:17232] = np.where(matrix[0:17232,0:17232] > 0,  10,  0) 
        matrix[0:17232,17232:] = np.where(matrix[0:17232,17232:] > 0,  4,  0) 
        matrix[17232:,0:17232] = np.where(matrix[17232:,0:17232] > 0,  4,  0) 
        matrix[17232:,17232:] = np.where(matrix[17232:,17232:] > 0, -10,  0) 
        #plt.imsave(save_path+'/'+tr_type+'_'+str(id_part)+'_'+str(THRESHOLD)+'.png',matrix,vmax=10, vmin=-10, cmap ='coolwarm')
        plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'.jpeg',\
            matrix, vmax=vmax, vmin=vmin, cmap ='seismic')  #coolwarm
        plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_gryi.jpeg',\
            matrix[0:17232,0:17232], vmax=vmax, vmin=vmin, cmap ='seismic')
        plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_sulci.jpeg',\
            matrix[17232:,17232:], vmax=vmax, vmin=vmin, cmap ='seismic')
        plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_gyri_sulci1.jpeg',\
            matrix[0:17232,17232:], vmax=5, vmin=-5, cmap ='PRGn')
        plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_gyri_sulci2.jpeg',\
            matrix[17232:,0:17232], vmax=5, vmin=-5, cmap ='PRGn')
        print('max value in subject signal ', subject_signal.max())
        signal  = np.where(signal > 0, 1, signal)
        signal  = np.where(signal < 0, -1, signal)
        #write this signal 
        scalars.append(signal)
        labels.append("label{}".format(id_part))

        rewrite_scalars("/media/shawey/SSD8T/GyraSulci_Motor/InflatedSurface/InflatedSurface.vtk", \
                    save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+"_zscore_Inflated.vtk",new_scalars=scalars,new_scalar_names=labels)

        scalars = []
        labels = []
