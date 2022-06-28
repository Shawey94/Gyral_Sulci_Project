import torch, vtk
import numpy as np
import h5py
from vtk_utils import *
import os
from scipy.stats import zscore

path = 'Twin_Transformers'
folder_name = '/media/shawey/SSD8T/GyraSulci_Motor/'+path+'/figs'
weis = os.listdir(folder_name)

filename = "/media/shawey/SSD8T/GyraSulci_Motor/H5_vtk/100408.h5"
f1 = h5py.File(filename,'r+')    

roi = f1['roi']
len(roi[roi[:] == True])

reco_subs = []
for wei in weis:
    print(wei)
    sub_id = wei.split('_')[0]
    if sub_id in reco_subs:
        continue
    reco_subs.append(sub_id)
    ws = [sub_id + '_gyri_weight.pkl',sub_id + '_sulci_weight.pkl' ]
    #write this signal 
    for w in ws:
        part = w.split('_')[1]
        scalars = []
        labels = []
        #w1 = torch.load(folder_name+"/" + w).cpu().numpy()  #122317_gyri_weight.pkl
        #w1 = zscore(w1)      #!!! WITH OR WITHOUT zscore
        temp = torch.load(folder_name+"/" + w,torch.device("cpu")).t()
        w1 = zscore(temp)[:59412,:]
        w1 = np.transpose(w1)
        for i in range(100):
            signal = np.zeros(64984)
            signal[roi[:] == True] = w1[i][:59412]

            #write this signal 
            scalars.append(signal)
            labels.append("label{}".format(i))

        rewrite_scalars("/media/shawey/SSD8T/GyraSulci_Motor/InflatedSurface/InflatedSurface.vtk", \
                    "/media/shawey/SSD8T/GyraSulci_Motor/"+path+"/generated_vtks/"+\
                                sub_id+"_"+part+"_zscore.vtk",new_scalars=scalars,new_scalar_names=labels)
    #break

