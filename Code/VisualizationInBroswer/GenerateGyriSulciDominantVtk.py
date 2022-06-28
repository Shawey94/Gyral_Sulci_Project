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

def ConstructGraph(gyri_network, sulci_network,gyri_signal,sulci_signal, gyri_pos, sulci_pos, roi):
    gyri_sulci_network = np.concatenate((gyri_network, sulci_network))
    gyri_sulci_indices = np.array(np.where(gyri_sulci_network == 1.0)).reshape(1,-1)   # indices <= 35559
    
    #--------------------------------------------------------------------------------
    all_signal = -1 * np.ones(64984) 
    roi_signal = -1 * np.ones(59412)
    
    for pos in gyri_sulci_indices[0,:]:            
        if(pos < 17232):   #17232 gyri
            roi_signal[gyri_pos[0,pos]] = 0.7 #abs(gyri_signal[pos])
        elif(pos < 35559):
            #print(sulci_pos[0,pos-17232])
            roi_signal[sulci_pos[0,pos-17232]] = -0.7 #abs(sulci_signal[pos-17232])        
    all_signal[roi[:] == True] = roi_signal           
    #---------------------------------------------------------------------------------

    return all_signal

def LoadParams():
    filename = "/media/shawey/SSD8T/GyraSulci_Motor/H5_vtk/100408.h5"
    f1 = h5py.File(filename,'r+')
    roi = f1['roi']
    label = ReadLabel('100408')
    mask_label = label[roi[:]==True]
    return roi, mask_label

def main(path, pkl_name, save_path, id_part,THRESHOLD,component_id):
    w1 = zscore(torch.load(path+pkl_name,torch.device("cpu")))[:,:59412]
    roi, mask_label = LoadParams()

    tick1 = time.time()
    print(i)
    gyri_network = w1[component_id][mask_label==1]                  #gyri 17232
    gyri_signal = gyri_network
    gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
    # this is in mask_label level (including gyri, sulci and unknown),not in roi level
    gyri_network = np.where(gyri_network >= THRESHOLD,  gyri_network, 0)
    gyri_network = np.where(gyri_network < THRESHOLD, gyri_network, 1 )  

    sulci_network = w1[component_id][mask_label==-1]    #sulci 18327
    sulci_signal = sulci_network
    sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 
    sulci_network = np.where(sulci_network >= THRESHOLD,  sulci_network, 0)
    sulci_network = np.where(sulci_network < THRESHOLD, sulci_network, 1 )  
    signal= ConstructGraph(gyri_network, sulci_network,gyri_signal,sulci_signal, gyri_pos, sulci_pos,roi)

    tick2 = time.time()
    print('time usage ', tick2-tick1)
    return signal
    
#USE gyri_weight generated from haxing
path = "/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/figs/"
pkls = os.listdir(path)
save_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/GyriSulciDominantVtks'
THRESHOLDs = [2,1.5]

for THRESHOLD in THRESHOLDs:
    reco_subs = []
    print('threshold is ', THRESHOLD)
    for pkl in pkls:
        print(pkl)
        sub_id = pkl.split('_')[0]
        if sub_id in reco_subs:
            continue
        reco_subs.append(sub_id)
        #ws = [sub_id + '_gyri_weight.pkl',sub_id + '_sulci_weight.pkl' ]
        part = pkl.split('_')[1]
        id_part = sub_id + '_' + part
        scalars = []
        labels = []
        for i in range(100):
            if (i > 85 or i < 10):  # common spa or common tem
                subject_signal = main(path,pkl,save_path,id_part,THRESHOLD/2, i) # i is component id

                print('max value in subject signal ', abs(subject_signal).max())
                #subject_signal = subject_signal/ abs(subject_signal).max()
                #write this signal 
                scalars.append(subject_signal)
                labels.append("label{}".format(i))

            else:
                subject_signal = main(path,pkl,save_path,id_part,THRESHOLD, i) # i is component id
                print('max value in subject signal ', abs(subject_signal).max())
                #subject_signal = subject_signal/ abs(subject_signal).max()
                #write this signal 
                scalars.append(subject_signal)
                labels.append("label{}".format(i))

        rewrite_scalars("/media/shawey/SSD8T/GyraSulci_Motor/InflatedSurface/InflatedSurface.vtk", \
                    save_path+'/'+sub_id+"_"+part+'_'+str(THRESHOLD)+"_Dominant_zscore.vtk",new_scalars=scalars,new_scalar_names=labels)

