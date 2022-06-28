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
print(len(roi[roi[:] == True]))

reco_subs = []
for wei in weis:
    w1 = torch.load(folder_name+"/" + wei).cpu().numpy()  #122317_gyri_weight.pkl
    print(wei)
    sub_id = wei.split('_')[0]
    if sub_id in reco_subs:
        continue
    reco_subs.append(sub_id)
    if(sub_id == '131722'):
        Jaccard = np.zeros(100)
        ws = [sub_id + '_gyri_weight.pkl',sub_id + '_sulci_weight.pkl' ]
        #write this signal 
        gyri = torch.load(folder_name+"/" + ws[0]).cpu().numpy()  #122317_gyri_weight.pkl
        sulci = torch.load(folder_name+"/" + ws[1]).cpu().numpy()  #122317_gyri_weight.pkl
        gyri = zscore(gyri)
        sulci = zscore(sulci)
        for i in range(100):
            #print('pcc ', np.corrcoef(gyri[i][:59412], sulci[i][:59412])[0,1])
            gyri_signal = np.zeros(64984)
            gyri_signal[roi[:] == True] = gyri[i][:59412]
            thre = 0.05
            gyri_signal = np.where(gyri_signal >= thre,  gyri_signal, 0)
            gyri_signal = np.where(gyri_signal < thre, gyri_signal, 1 )  

            sulci_signal = np.zeros(64984)
            sulci_signal[roi[:] == True] = sulci[i][:59412]
            thre = 0.05 #sulci_signal.max() * 0.05
            sulci_signal = np.where(sulci_signal >= thre,  sulci_signal, 0)
            sulci_signal = np.where(sulci_signal < thre, sulci_signal, 1 )
            #sulci_num = np.sum(sulci_signal == 1 )    

            intersect = 0 
            union = 0
            for k in range(64984):
                if(sulci_signal[k] == gyri_signal[k] and gyri_signal[k] == 1.0):
                    intersect += 1
                    union += 1
                    continue
                if(sulci_signal[k] == 1 or gyri_signal[k] == 1):
                    union += 1
            Jaccard[i]= intersect / union

print(Jaccard)
print(Jaccard.mean())
print(Jaccard.max())
print(list(Jaccard).index(Jaccard.max()))
