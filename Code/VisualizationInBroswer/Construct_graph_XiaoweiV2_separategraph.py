#CONSTRCT CorePeriphery Matrix
import torch
import numpy as np
from scipy.stats import zscore
import h5py    
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
import scipy.sparse as sp

def ReadLabel(sub_id):
  vtk_filename =  "/media/shawey/SSD8T/GyraSulci_Motor/H5_vtk/{}.vtk".format(sub_id)
  flag = 0 
  label = []
  for line in open(vtk_filename):
    if "LOOKUP_TABLE gyri_sulci_roi" in line:
      flag = 1
      continue
    if flag == 0:
      continue
    label.append(int(float(line.strip())))
  return np.array(label)

def ConstructGraph(gyri_network, sulci_network):
          whole_network = np.concatenate((gyri_network, sulci_network))
          indices = np.array(np.where(whole_network == 1.0)).reshape(1,-1)
          matrix = np.zeros((35559,35559) ) #first 17232 gyri, 18327 sulci
          #dtype='i1'
          for i in indices[0,:]:
                    for j in indices[0,:]:
                              matrix[i,j] = 1
          return matrix 

def LoadParams():
          filename = "/media/shawey/SSD8T/GyraSulci_Motor/H5_vtk/100408.h5"
          f1 = h5py.File(filename,'r+')    
          roi = f1['roi']
          label = ReadLabel("100408")
          mask_label = label[roi[:]==True]
          return roi, mask_label


def WritePartial2vtk(subject_signal, flag):
    subject_signal = np.where(subject_signal > 0, 1, subject_signal)
    subject_signal = np.where(subject_signal < 0, -1, subject_signal)
    scalars = []
    labels = []
    #write this signal 
    scalars.append(subject_signal)
    labels.append("label{}".format(0))

    rewrite_scalars("/media/shawey/SSD8T/GyraSulci_Motor/InflatedSurface/InflatedSurface.vtk", \
            save_path+'/PartitionMatrix2Vtks/'+sub_id+"_"+part+'_'+str(THRESHOLD)+'_'+str(flag)+"_zscore_Inflated.vtk",new_scalars=scalars,new_scalar_names=labels)

def PartitionSubjectMatrix(subject_matrix,  gyri_pos, sulci_pos, roi):
    all_signal = np.zeros(64984)
    cnt = 0
    distance = 0
    temp_signal = np.zeros(59412)
    indics_reco = []
    for j in range(17232):
        if( (subject_matrix[0:17232, j]).max() > 0 ):
            temp_signal[gyri_pos[0,j]] = 1 #(subject_matrix[0:17232, j]).max()
            cnt += 1
            indics_reco.append(j)
            flag = j
        else:
            if(cnt > 0):
                distance += 1
            if(distance >= 1000 and cnt > 100):
                distance = 0
                all_signal[roi[:] == True] = temp_signal  
                #WritePartial2vtk(all_signal, 'gyri_'+str(cnt)+'_'+str(flag)) 
                temp_signal = np.zeros(59412)
                cnt = 0
            elif(distance >= 1000 and cnt < 100 ):
                distance = 0
                cnt = 0
    print(indics_reco)
    print(len(indics_reco))

    patience = 0
    cnt = 0
    all_signal = np.zeros(64984)
    temp_signal = np.zeros(59412) 
    indics_reco = []
    for j in range(17232, 35559):  #sulci 18327
        #print('j {}, max {}'.format(j, (subject_matrix[:, j]).max()))
        if( (subject_matrix[:, j]).max() > 0 ):
            temp_signal[sulci_pos[0,j-17232]] = -1 #(subject_matrix[17232:, j]).max()
            indics_reco.append(j)
            cnt += 1
            flag = j
        else:
            if(cnt > 0):
                patience += 1
            if(patience >= 600 and cnt > 20 ): #and cnt > 100
                patience = 0
                all_signal[roi[:] == True] = temp_signal  
                #WritePartial2vtk(all_signal, 'sulci_'+str(cnt)+'_'+str(flag)) 
                temp_signal = np.zeros(59412)
                cnt = 0
            elif(patience >= 600 and cnt < 20 ):
                patience = 0
                cnt = 0
    print(indics_reco)
    print(len(indics_reco))

def main(path, pkl_name, save_path, id_part, tr_type,THRESHOLD):
          #load weight 
          w1 = zscore(torch.load(path+pkl_name,torch.device("cpu")))[:,:59412]
          roi, mask_label = LoadParams()
          gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
          sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 
          #constructive graph
          #print(w1[0][mask_label==1][:20])
          matrix = np.zeros((35559,35559))  #dtype='i1'

          if tr_type == "full":
                    tr_index = range(100)
          elif tr_type == "common_spa":
                    tr_index = range(15)
          elif tr_type == "common_tem":
                    tr_index = range(15,30)
          elif tr_type == "indv":
                    tr_index = range(30,100)
          tick1 = time.time()
          for i in tr_index:
                    print(i)
                    gyri_network = w1[i][mask_label==1]                  #17232
                    gyri_network = np.where(gyri_network >= THRESHOLD,  gyri_network, 0)
                    gyri_network = np.where(gyri_network < THRESHOLD, gyri_network, 1 )  
                    sulci_network = w1[i][mask_label==-1]                #18327
                    sulci_network = np.where(sulci_network >= THRESHOLD,  sulci_network, 0)
                    sulci_network = np.where(sulci_network < THRESHOLD, sulci_network, 1 )  
                    matrix += ConstructGraph(gyri_network, sulci_network)

          #PartitionSubjectMatrix(matrix, gyri_pos, sulci_pos, roi)
          #sp_matrix = sp.csr_matrix(matrix)
          #sp.save_npz(save_path+'/sparse_matrix.npz',sp_matrix)
          vmin = -20
          vmax = 20
          matrix[0:17232,0:17232] = np.where(matrix[0:17232,0:17232] > 0,  10,  0) 
          matrix[0:17232,17232:] = np.where(matrix[0:17232,17232:] > 0,  15,  0) 
          matrix[17232:,0:17232] = np.where(matrix[17232:,0:17232] > 0,  15,  0) 
          matrix[17232:,17232:] = np.where(matrix[17232:,17232:] > 0, -10,  0) 
          plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'.jpeg',\
            matrix, vmax=vmax, vmin=vmin, cmap ='seismic')  #coolwarm
          plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_gryi.jpeg',\
            matrix[0:17232,0:17232], vmax=vmax, vmin=vmin, cmap ='seismic')
          plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_sulci.jpeg',\
            matrix[17232:,17232:], vmax=vmax, vmin=vmin, cmap ='seismic')
          plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_gyri_sulci1.jpeg',\
            matrix[0:17232,17232:], vmax=vmax, vmin=vmin, cmap ='PRGn')
          plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'_gyri_sulci2.jpeg',\
            matrix[17232:,0:17232], vmax=vmax, vmin=vmin, cmap ='PRGn')
          print("matrix max",np.max(matrix))
          print("matrix min",np.min(matrix))
          tick2 = time.time()
          print('time usage ', tick2-tick1)

path = "/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/CorePeriphery/"
pkls = os.listdir(path)
save_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/CorePeriphery'
THRESHOLD = 2.5
for pkl in pkls:
    print(pkl)
    if(pkl == 'gyri_weight.pkl'): 
       sub_id = pkl.split('_')[0]
       part = pkl.split('_')[1]
       id_part = sub_id + '_' + part
       which_comps = 'common_spa'
       main(path,pkl,save_path,id_part,which_comps, THRESHOLD)

print('end')
