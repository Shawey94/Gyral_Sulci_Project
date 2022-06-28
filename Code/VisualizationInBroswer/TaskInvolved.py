import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import cv2

def WholeRatio(temporal_path, comp_num, pcc_thre):
    basises = os.listdir(temporal_path)

    label_path = '/media/shawey/SSD8T/HCP/HCP3T4mm/data_code_bm_template_task_label/MOTOR_label.mat'
    data = scio.loadmat(label_path)
    #print(data)
    label = data['Label']
    print(label.shape)
    time_point, num_type = label.shape   
    
    gyri_task_par_cnt = 0 
    sulci_task_par_cnt = 0 
    reco_subs = []
    for basis in basises:
        sub_id = basis.split('_')[0]
        print(sub_id)
        if (sub_id in reco_subs):
            continue
        reco_subs.append(sub_id)
        part_id = basis.split('_')[1]
        comp_id = basis.split('_')[2]
        print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))

        for comp in range(comp_num):
            gyri_signal = np.loadtxt(temporal_path+'/'+sub_id + '_gyri_' + str(comp) +'_temporal_basis.txt')  #166438_gyri_2_temporal_basis.txt
            sulci_signal = np.loadtxt( temporal_path+'/'+sub_id + '_sulci_' + str(comp) +'_temporal_basis.txt' )
            gyri_signal = gyri_signal / max(abs(gyri_signal))
            sulci_signal = sulci_signal / max(abs(sulci_signal))
            
            for i in range(5):
                temp =  label[:, i]
                gyri_pcc = int(np.corrcoef(temp, gyri_signal)[0,1] * 1000) / 1000
                if ( abs(gyri_pcc) > pcc_thre):
                    print('gyri num_type is ', i)      
                    print('gyri pcc is ', gyri_pcc ) 
                    gyri_task_par_cnt += 1           
            
            for i in range(5):
                temp =  label[:, i]
                sulci_pcc = int(np.corrcoef(temp, sulci_signal)[0,1] * 1000) / 1000
                if ( abs(sulci_pcc) > pcc_thre):
                    print('sulci num_type is ', i)      
                    print('sulci pcc is ', sulci_pcc ) 
                    sulci_task_par_cnt += 1

            #pcc = np.corrcoef(gyri_signal, sulci_signal)[0,1]
            #pcc = int(pcc * 1000) / 1000

    return gyri_task_par_cnt / 5/comp_num / len(reco_subs), sulci_task_par_cnt/ 5/comp_num / len(reco_subs)

path = 'Twin_Transformers'
temporal_path = '/media/shawey/SSD8T/GyraSulci_Motor/'+path+'/temporal_basis'
gyri_task_ratio, sulci_task_ratio = WholeRatio(temporal_path, 100, 0.1)

print('whole: gyri_task_ratio {}, sulci_task_ratio {}'.format(gyri_task_ratio, sulci_task_ratio))  #100 comps * 5 tasks



def CommonSpaRatio(temporal_path, comp_num, commspa_num, pcc_thre):
    basises = os.listdir(temporal_path)

    label_path = '/media/shawey/SSD8T/HCP/HCP3T4mm/data_code_bm_template_task_label/MOTOR_label.mat'
    data = scio.loadmat(label_path)
    #print(data)
    label = data['Label']
    #print(label.shape)
    time_point, num_type = label.shape   
    
    gyri_task_par_cnt = 0 
    sulci_task_par_cnt = 0 
    reco_subs = []
    for basis in basises:
        sub_id = basis.split('_')[0]
        #print(sub_id)
        if (sub_id in reco_subs):
            continue
        reco_subs.append(sub_id)
        part_id = basis.split('_')[1]
        comp_id = basis.split('_')[2]
        print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))

        for comp in range(comp_num-commspa_num, comp_num, 1):
            gyri_signal = np.loadtxt(temporal_path+'/'+sub_id + '_gyri_' + str(comp) +'_temporal_basis.txt')  #166438_gyri_2_temporal_basis.txt
            sulci_signal = np.loadtxt( temporal_path+'/'+sub_id + '_sulci_' + str(comp) +'_temporal_basis.txt' )
            gyri_signal = gyri_signal / max(abs(gyri_signal))
            sulci_signal = sulci_signal / max(abs(sulci_signal))
            
            for i in range(5):
                temp =  label[:, i]
                gyri_pcc = int(np.corrcoef(temp, gyri_signal)[0,1] * 1000) / 1000
                if ( abs(gyri_pcc) > pcc_thre):
                    #print('gyri num_type is ', i)      
                    #print('gyri pcc is ', gyri_pcc ) 
                    gyri_task_par_cnt += 1           
            
            for i in range(5):
                temp =  label[:, i]
                sulci_pcc = int(np.corrcoef(temp, sulci_signal)[0,1] * 1000) / 1000
                if ( abs(sulci_pcc) > pcc_thre):
                    #print('sulci num_type is ', i)      
                    #print('sulci pcc is ', sulci_pcc ) 
                    sulci_task_par_cnt += 1

            #pcc = np.corrcoef(gyri_signal, sulci_signal)[0,1]
            #pcc = int(pcc * 1000) / 1000

    return gyri_task_par_cnt / 5/commspa_num / len(reco_subs), sulci_task_par_cnt/ 5/commspa_num / len(reco_subs)

gyri_task_ratio, sulci_task_ratio = CommonSpaRatio(temporal_path, 100, 15, 0.1) #last 15 rows

print('CommSpa: gyri_task_ratio {}, sulci_task_ratio {}'.format(gyri_task_ratio, sulci_task_ratio))  #100 comps * 5 tasks


def CommonTemRatio(temporal_path, comp_num, commtem_num, pcc_thre):
    basises = os.listdir(temporal_path)

    label_path = '/media/shawey/SSD8T/HCP/HCP3T4mm/data_code_bm_template_task_label/MOTOR_label.mat'
    data = scio.loadmat(label_path)
    #print(data)
    label = data['Label']
    #print(label.shape)
    time_point, num_type = label.shape   
    
    gyri_task_par_cnt = 0 
    sulci_task_par_cnt = 0 
    reco_subs = []
    for basis in basises:
        sub_id = basis.split('_')[0]
        #print(sub_id)
        if (sub_id in reco_subs):
            continue
        reco_subs.append(sub_id)
        part_id = basis.split('_')[1]
        comp_id = basis.split('_')[2]
        print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))

        for comp in range(0, commtem_num, 1):
            gyri_signal = np.loadtxt(temporal_path+'/'+sub_id + '_gyri_' + str(comp) +'_temporal_basis.txt')  #166438_gyri_2_temporal_basis.txt
            sulci_signal = np.loadtxt( temporal_path+'/'+sub_id + '_sulci_' + str(comp) +'_temporal_basis.txt' )
            gyri_signal = gyri_signal / max(abs(gyri_signal))
            sulci_signal = sulci_signal / max(abs(sulci_signal))
            
            for i in range(5):
                temp =  label[:, i]
                gyri_pcc = int(np.corrcoef(temp, gyri_signal)[0,1] * 1000) / 1000
                if ( abs(gyri_pcc) > pcc_thre):
                    #print('gyri num_type is ', i)      
                    #print('gyri pcc is ', gyri_pcc ) 
                    gyri_task_par_cnt += 1           
            
            for i in range(5):
                temp =  label[:, i]
                sulci_pcc = int(np.corrcoef(temp, sulci_signal)[0,1] * 1000) / 1000
                if ( abs(sulci_pcc) > pcc_thre):
                    #print('sulci num_type is ', i)      
                    #print('sulci pcc is ', sulci_pcc ) 
                    sulci_task_par_cnt += 1

            #pcc = np.corrcoef(gyri_signal, sulci_signal)[0,1]
            #pcc = int(pcc * 1000) / 1000

    return gyri_task_par_cnt / 5/commtem_num / len(reco_subs), sulci_task_par_cnt/ 5/commtem_num / len(reco_subs)

gyri_task_ratio, sulci_task_ratio = CommonTemRatio(temporal_path, 100, 10, 0.1) #first 10 columns

print('CommTem: gyri_task_ratio {}, sulci_task_ratio {}'.format(gyri_task_ratio, sulci_task_ratio))  #100 comps * 5 tasks


def IndiRatio(temporal_path, comp_num, commtem_num, commspa_num, pcc_thre):
    basises = os.listdir(temporal_path)

    label_path = '/media/shawey/SSD8T/HCP/HCP3T4mm/data_code_bm_template_task_label/MOTOR_label.mat'
    data = scio.loadmat(label_path)
    #print(data)
    label = data['Label']
    #print(label.shape)
    time_point, num_type = label.shape   
    
    gyri_task_par_cnt = 0 
    sulci_task_par_cnt = 0 
    reco_subs = []
    for basis in basises:
        sub_id = basis.split('_')[0]
        #print(sub_id)
        if (sub_id in reco_subs):
            continue
        reco_subs.append(sub_id)
        part_id = basis.split('_')[1]
        comp_id = basis.split('_')[2]
        #print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))

        for comp in range(commtem_num, comp_num - commspa_num, 1):
            gyri_signal = np.loadtxt(temporal_path+'/'+sub_id + '_gyri_' + str(comp) +'_temporal_basis.txt')  #166438_gyri_2_temporal_basis.txt
            sulci_signal = np.loadtxt( temporal_path+'/'+sub_id + '_sulci_' + str(comp) +'_temporal_basis.txt' )
            gyri_signal = gyri_signal / max(abs(gyri_signal))
            sulci_signal = sulci_signal / max(abs(sulci_signal))
            
            for i in range(5):
                temp =  label[:, i]
                gyri_pcc = int(np.corrcoef(temp, gyri_signal)[0,1] * 1000) / 1000
                if ( abs(gyri_pcc) > pcc_thre):
                    #print('gyri num_type is ', i)      
                    #print('gyri pcc is ', gyri_pcc ) 
                    gyri_task_par_cnt += 1           
            
            for i in range(5):
                temp =  label[:, i]
                sulci_pcc = int(np.corrcoef(temp, sulci_signal)[0,1] * 1000) / 1000
                if ( abs(sulci_pcc) > pcc_thre):
                    #print('sulci num_type is ', i)      
                    #print('sulci pcc is ', sulci_pcc ) 
                    sulci_task_par_cnt += 1

            #pcc = np.corrcoef(gyri_signal, sulci_signal)[0,1]
            #pcc = int(pcc * 1000) / 1000

    return gyri_task_par_cnt / 5/( comp_num - commspa_num - commtem_num) / len(reco_subs), sulci_task_par_cnt/ 5/( comp_num - commspa_num - commtem_num) / len(reco_subs)

gyri_task_ratio, sulci_task_ratio = IndiRatio(temporal_path, 100, 10, 15, 0.25) #first 10 columns

print('Indi: gyri_task_ratio {}, sulci_task_ratio {}'.format(gyri_task_ratio, sulci_task_ratio))  #100 comps * 5 tasks
