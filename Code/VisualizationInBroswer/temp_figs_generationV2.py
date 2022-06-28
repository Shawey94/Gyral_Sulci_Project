import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import cv2

def temporal_pattern_fig_generation(temporal_path, save_fig_path):
    basises = os.listdir(temporal_path)

    label_path = '/media/shawey/SSD8T/HCP/HCP3T4mm/data_code_bm_template_task_label/MOTOR_label.mat'
    data = scio.loadmat(label_path)
    #print(data)
    label = data['Label']
    print(label.shape)
    time_point, num_type = label.shape   

    for basis in basises:
        sub_id = basis.split('_')[0]
        part_id = basis.split('_')[1]
        comp_id = basis.split('_')[2]
        print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))
        gyri_signal = np.loadtxt(temporal_path+'/'+sub_id + '_gyri_' + comp_id +'_temporal_basis.txt')  #166438_gyri_2_temporal_basis.txt
        sulci_signal = np.loadtxt( temporal_path+'/'+sub_id + '_sulci_' + comp_id +'_temporal_basis.txt' )
        gyri_signal = gyri_signal / max(abs(gyri_signal))
        sulci_signal = sulci_signal / max(abs(sulci_signal))
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(gyri_signal, color = 'r',  linewidth = 1, label = sub_id + '_gyri_' + comp_id  )   #'#ffb07c' 
        
        for i in range(5):
            temp =  label[:, i]
            gyri_pcc = int(np.corrcoef(temp, gyri_signal)[0,1] * 1000) / 1000
            if ( abs(gyri_pcc) > 0.2):
                if(i == 0):
                    ax1.plot(temp, color = 'gold', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(gyri_pcc))
                if(i == 1):
                    ax1.plot(temp, color = 'g', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(gyri_pcc))
                if(i == 2):
                    ax1.plot(temp, color = 'y', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(gyri_pcc))
                if(i == 3):
                    ax1.plot(temp, color = 'k', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(gyri_pcc))
                if(i == 4):
                    ax1.plot(temp, color = 'm', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(gyri_pcc))
                print('gyri num_type is ', i)      
                print('gyri pcc is ', gyri_pcc ) 
        
        ax1.axis('off')
        ax1.legend( fontsize = 10)
        ax2.plot(sulci_signal, color = 'b',  linewidth = 1, label = sub_id + '_sulci_' + comp_id  )   #'#ffb07c' 
        
        for i in range(5):
            temp =  label[:, i]
            sulci_pcc = int(np.corrcoef(temp, sulci_signal)[0,1] * 1000) / 1000
            if ( abs(sulci_pcc) > 0.2):
                if(i == 0):
                    ax2.plot(temp, color = 'gold', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(sulci_pcc))
                if(i == 1):
                    ax2.plot(temp, color = 'g', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(sulci_pcc))
                if(i == 2):
                    ax2.plot(temp, color = 'y', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(sulci_pcc))
                if(i == 3):
                    ax2.plot(temp, color = 'k', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(sulci_pcc))
                if(i == 4):
                    ax2.plot(temp, color = 'm', linewidth = 1 , label = 'TD_' + str(i) + '_PCC ' + str(sulci_pcc))
                print('sulci num_type is ', i)      
                print('sulci pcc is ', sulci_pcc ) 

        ax2.axis('off')
        ax2.legend( fontsize = 10)
        pcc = np.corrcoef(gyri_signal, sulci_signal)[0,1]
        pcc = int(pcc * 1000) / 1000
        plt.title('gyri sulci signal pcc '+ str(pcc), fontsize = 10)
        #plt.show()
        fig.savefig(save_fig_path+'/'+str(sub_id)+'_'+str(comp_id)+'_gyri_sulci.png') #166438_1_gyri_sulci.png
        plt.close()

path = 'Twin_Transformers'
temporal_path = '/media/shawey/SSD8T/GyraSulci_Motor/'+path+'/temporal_basis'
temp_save_path = '/media/shawey/SSD8T/GyraSulci_Motor/'+path+'/VisualizationInBroswer/temporal_pngs'
temporal_pattern_fig_generation(temporal_path, temp_save_path )






