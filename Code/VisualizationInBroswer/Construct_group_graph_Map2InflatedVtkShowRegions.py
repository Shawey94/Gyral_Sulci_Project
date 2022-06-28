#show small parts of matrix to surface
#save matrix jepg
#show whole brain
import torch
import numpy as np
from scipy.stats import zscore
import h5py
import matplotlib.pyplot as plt
import time
import os
from vtk_utils import *
import copy

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

def main(path, pkl_name, save_path, id_part, tr_type,THRESHOLD):
    temp = torch.load(path+pkl_name,torch.device("cpu")).t()
    w1 = zscore(temp)[:59412,:]
    w1 = np.transpose(w1)
    #w1 = zscore(torch.load(path+pkl_name,torch.device("cpu")))[:,:59412]
    roi, mask_label = LoadParams()
    matrix = np.zeros((35559,35559),dtype='i1')
    signal = np.zeros(64984)

    if tr_type == "full":
       tr_index = range(400)
    elif tr_type == "comm_spa":
       tr_index = range(350,400)
    elif tr_type == "comm_tem":
       tr_index = range(0,50)
    elif tr_type == "indv":
       tr_index = range(50,350)

    tick1 = time.time()
    gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
    sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 
    for i in tr_index:
        print(i)
        gyri_network = w1[i][mask_label==1]                  #gyri 17232
        gyri_signal = gyri_network
        # this is in mask_label level (including gyri, sulci and unknown),not in roi level
        gyri_network = np.where(gyri_network >= THRESHOLD,  gyri_network, 0)
        gyri_network = np.where(gyri_network < THRESHOLD, gyri_network, 1 )  

        unknown_network = w1[i][mask_label==0]                  
        unknown_pos = np.array(np.where(mask_label == 0)).reshape(1,-1) 
        # this is in mask_label level (including gyri, sulci and unknown),not in roi level
        unknown_network = np.where(unknown_network >= 10000,  unknown_network, 0)

        sulci_network = w1[i][mask_label==-1]    #sulci 18327
        sulci_signal = sulci_network
        sulci_network = np.where(sulci_network >= THRESHOLD,  sulci_network, 0)
        sulci_network = np.where(sulci_network < THRESHOLD, sulci_network, 1 )  
        temp_matrix, temp_signal= ConstructGraph(gyri_network, sulci_network,unknown_network,gyri_signal,sulci_signal, gyri_pos, sulci_pos,roi, mask_label)
  
        matrix += temp_matrix
        signal += temp_signal
    tick2 = time.time()
    print('time usage ', tick2-tick1)
    print("matrix max",np.max(matrix))
    #plt.imsave(save_path+'/'+id_part+'_'+tr_type+'_'+str(THRESHOLD)+'.jpeg',\
    #        matrix, vmax=5, vmin=-5, cmap ='coolwarm')
    return matrix, signal


def WritePartial2vtk(subject_signal, flag):
    subject_signal = np.where(subject_signal > 0, 1, subject_signal)
    subject_signal = np.where(subject_signal < 0, -1, subject_signal)
    scalars = []
    labels = []
    #write this signal 
    scalars.append(subject_signal)
    labels.append("label{}".format(0))

    rewrite_scalars("/media/shawey/SSD8T/GyraSulci_Motor/InflatedSurface/InflatedSurface.vtk", \
            save_path+'/'+str(THRESHOLD)+'_'+str(flag)+"_zscore_Inflated.vtk",new_scalars=scalars,new_scalar_names=labels)

def PartitionSubjectMatrix(subject_matrix,  gyri_pos, sulci_pos):
    '''
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
            if(distance >= 700 and cnt > 100):
                distance = 0
                all_signal[roi[:] == True] = temp_signal  
                WritePartial2vtk(all_signal, 'gyri_'+str(cnt)+'_'+str(flag)) 
                temp_signal = np.zeros(59412)
                cnt = 0
            elif(distance >= 700 and cnt < 100 ):
                distance = 0
                cnt = 0
    print(indics_reco)
    print(len(indics_reco))
    '''

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
            if(patience >= 400 and cnt > 40 ): #and cnt > 100
                patience = 0
                all_signal[roi[:] == True] = temp_signal  
                WritePartial2vtk(all_signal, 'sulci_'+str(cnt)+'_'+str(flag)) 
                temp_signal = np.zeros(59412)
                cnt = 0
            elif(patience >= 500 and cnt < 20 ):
                patience = 0
                cnt = 0
    print(indics_reco)
    print(len(indics_reco))

def PartitionSubjectMatrix_custom(subject_matrix,  gyri_pos, sulci_pos):
    gyri_all_signal_list = []
    gyri_all_signal = np.zeros(64984)
    temp_signal = np.zeros(59412)
    gyri_dic = {1:[5450,6563], 2:[11100,11884], 3:[14506,15213], 4:[16233,16707]}
    indics_reco = []
    cnt = 0
    for key in gyri_dic.keys():
        le = gyri_dic[key][0]
        ri = gyri_dic[key][1]
        for i in range(le, ri):
            if( (subject_matrix[:, i]).max() > 0 ):
                cnt += 1
                temp_signal[gyri_pos[0,i]] = 1 #(subject_matrix[17232:, j]).max()
        gyri_all_signal[roi[:] == True] = temp_signal  
        #WritePartial2vtk(gyri_all_signal, 'gyri_'+str(cnt)+'_'+str(ri)) 
        cnt = 0
        temp_signal = np.zeros(59412)
        gyri_all_signal_list.append( copy.deepcopy(gyri_all_signal))
        print( abs((gyri_all_signal_list[0] - gyri_all_signal)).max() )
    
    indics_reco = []
    for j in range(0, 17232):  #sulci 18327
        #print('j {}, max {}'.format(j, (subject_matrix[:, j]).max()))
        if( (subject_matrix[:, j]).max() > 0 ):
            indics_reco.append(j)
    print(indics_reco)
    print(len(indics_reco))
   
 
    indics_reco = []
    for j in range(17232, 35559):  #sulci 18327
        #print('j {}, max {}'.format(j, (subject_matrix[:, j]).max()))
        if( (subject_matrix[:, j]).max() > 0 ):
            indics_reco.append(j)
    print(indics_reco)
    print(len(indics_reco))

    sulci_all_signal_list = []
    sulci_all_signal = np.zeros(64984)
    temp_signal = np.zeros(59412)
    sulci_dic = { 1:[23494,23617], 2:[24159,24449], 3:[28000,28300],4:[31010,31212],\
       5:[33146,33363]}
    indics_reco = []
    cnt = 0
    for key in sulci_dic.keys():
        le = sulci_dic[key][0]
        ri = sulci_dic[key][1]
        for i in range(le, ri):
            if( (subject_matrix[:, i]).max() > 0 ):
                cnt += 1
                temp_signal[sulci_pos[0,i-17232]] = 1 #(subject_matrix[17232:, j]).max()
        sulci_all_signal[roi[:] == True] = temp_signal  
        #WritePartial2vtk(sulci_all_signal, 'sulci_'+str(cnt)+'_'+str(ri)) 
        cnt = 0
        temp_signal = np.zeros(59412)
        sulci_all_signal_list.append(copy.deepcopy(sulci_all_signal) )
    
    return gyri_all_signal_list, sulci_all_signal_list

    
#USE gyri_weight generated from haxing
path = "/media/shawey/SSD8T/GyraSulci_Motor/WM_task/TT2_CUDA/figs1/"
pkls = os.listdir(path)
save_path = '/media/shawey/SSD8T/GyraSulci_Motor/WM_task/TT2_CUDA/VisualizationInBroswer/gourp_wise_coreperiphery'
THRESHOLDs = [0.06]

roi, mask_label = LoadParams()
gyri_pos = np.array(np.where(mask_label == 1)).reshape(1,-1) 
sulci_pos = np.array(np.where(mask_label == -1)).reshape(1,-1) 

gyri_all_signal_list = []
sulci_all_signal_list = []
for i in range(4):
    gyri_all_signal_list.append(np.zeros(64984))
for i in range(5):
    sulci_all_signal_list.append(np.zeros(64984))

group_matrix = 0
for THRESHOLD in THRESHOLDs:
    print('threshold is ', THRESHOLD)
    for pkl in pkls:
        print(pkl)
        if( not os.path.isfile(path+pkl)):
            continue
        if( 'weight.pkl' in pkl):   
            sub_id = pkl.split('_')[0]
            part = pkl.split('_')[1]
            id_part = sub_id + '_' + part
            which_comps = 'comm_spa'
            subject_matrix, subject_signal = main(path,pkl,save_path,id_part,which_comps, THRESHOLD)
            print('max value in subject signal ', subject_signal.max())
            subject_signal  = np.where(subject_signal > 0, 1, subject_signal)
            subject_signal  = np.where(subject_signal < 0, -1, subject_signal)
            
            scalars = []
            labels = []
            #write this signal 
            scalars.append(subject_signal)
            labels.append("label{}".format(0))

            rewrite_scalars("/media/shawey/SSD8T/GyraSulci_Motor/InflatedSurface/InflatedSurface.vtk", \
                    save_path+'/'+sub_id+"_"+part+'_'+str(THRESHOLD)+"_zscore_Inflated.vtk",new_scalars=scalars,new_scalar_names=labels)
            
            sub_gyri_all_signal_list,sub_sulci_all_signal_list = PartitionSubjectMatrix_custom(subject_matrix, gyri_pos, sulci_pos)
        group_matrix += subject_matrix
        for i in range(4):
            gyri_all_signal_list[i] += sub_gyri_all_signal_list[i]

        for i in range(5):
            sulci_all_signal_list[i] += sub_sulci_all_signal_list[i]

    for i in range(4):
        WritePartial2vtk(gyri_all_signal_list[i], 'gyri_'+str(i)) 
    
    for i in range(5):
        WritePartial2vtk(sulci_all_signal_list[i], 'sulci_'+str(i)) 
    
    matrix = group_matrix
    vmin = -20
    vmax = 20
    matrix[0:17232,0:17232] = np.where(matrix[0:17232,0:17232] > 0,  10,  0) 
    matrix[0:17232,17232:] = np.where(matrix[0:17232,17232:] > 0,  4,  0) 
    matrix[17232:,0:17232] = np.where(matrix[17232:,0:17232] > 0,  4,  0) 
    matrix[17232:,17232:] = np.where(matrix[17232:,17232:] > 0, -10,  0) 
    #plt.imsave(save_path+'/'+tr_type+'_'+str(id_part)+'_'+str(THRESHOLD)+'.png',matrix,vmax=10, vmin=-10, cmap ='coolwarm')
    plt.imsave(save_path+'/'+str(THRESHOLD)+'.jpeg',\
            matrix, vmax=vmax, vmin=vmin, cmap ='seismic')  #coolwarm
    plt.imsave(save_path+'/'+str(THRESHOLD)+'_gryi.jpeg',\
            matrix[0:17232,0:17232], vmax=vmax, vmin=vmin, cmap ='seismic')
    plt.imsave(save_path+'/'+str(THRESHOLD)+'_sulci.jpeg',\
            matrix[17232:,17232:], vmax=vmax, vmin=vmin, cmap ='seismic')
    plt.imsave(save_path+'/'+str(THRESHOLD)+'_gyri_sulci1.jpeg',\
            matrix[0:17232,17232:], vmax=5, vmin=-5, cmap ='PRGn')
    plt.imsave(save_path+'/'+str(THRESHOLD)+'_gyri_sulci2.jpeg',\
            matrix[17232:,0:17232], vmax=5, vmin=-5, cmap ='PRGn')



