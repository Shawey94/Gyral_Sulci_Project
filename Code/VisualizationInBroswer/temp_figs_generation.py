import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import cv2

def temporal_pattern_fig_generation(temporal_path, save_fig_path):
    basises = os.listdir(temporal_path)

    for basis in basises:
        sub_id = basis.split('_')[0]
        part_id = basis.split('_')[1]
        comp_id = basis.split('_')[2]
        print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))
        #signal = np.loadtxt(temporal_path+'/'+basis)
        gyri_signal = np.loadtxt(temporal_path+'/'+sub_id + '_gyri_' + comp_id +'_temporal_basis.txt')  #166438_gyri_2_temporal_basis.txt
        sulcii_signal = np.loadtxt( temporal_path+'/'+sub_id + '_sulci_' + comp_id +'_temporal_basis.txt' )

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(gyri_signal, color = 'r',  linewidth = 1, label = sub_id + '_gyri_' + comp_id  )   #'#ffb07c' 
        ax1.axis('off')
        ax1.legend( fontsize = 20)
        ax2.plot(sulcii_signal, color = 'b',  linewidth = 1, label = sub_id + '_sulci_' + comp_id  )   #'#ffb07c' 
        ax2.axis('off')
        ax2.legend( fontsize = 20)
        pcc = np.corrcoef(gyri_signal, sulcii_signal)[0,1]
        pcc = int(pcc * 1000) / 1000
        plt.title(pcc, fontsize = 20)
        #plt.show()
        fig.savefig(save_fig_path+'/'+str(sub_id)+'_'+str(comp_id)+'_gyri_sulci.png') #166438_1_gyri_sulci.png
        plt.close()


temporal_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/temporal_basis'
temp_save_path = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/temporal_pngs'
temporal_pattern_fig_generation(temporal_path, temp_save_path )
gyri_sulci_figs = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/VisualizationInBroswer/static'

def gyri_sulci_temp_figs_generateion(temp_save_path, figs_path):
    figs = os.listdir(temp_save_path)

    for fig in figs:
        sub_id = fig.split('_')[0]   #101107_gyri30.png
        if( sub_id == '.DS'):
            continue
        part_id = fig.split('_')[1]
        comp_id = fig.split('_')[2].split('.')[0]
        print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))
        gyri_img = sub_id + '_gyri_' + comp_id +'.png'  #211215_sulci_90.png 
        sulci_img = sub_id + '_sulci_' + comp_id +'.png'
        img1 = cv2.imread( temp_save_path +'/'+ gyri_img )
        img2 = cv2.imread( temp_save_path +'/'+  sulci_img )
        #dim = ( img1.shape[1], int(img1.shape[0] /2))
        #img2 = cv2.resize(img2, dim) #interpolation= cv2.INTER_AREA
        gyri_sulci_fig = np.vstack((img1, img2))
        cv2.imwrite(figs_path +'/'+sub_id+'_'+comp_id+'_gyri_sulci.png', gyri_sulci_fig)

#gyri_sulci_temp_figs_generateion(temp_save_path, gyri_sulci_figs)






