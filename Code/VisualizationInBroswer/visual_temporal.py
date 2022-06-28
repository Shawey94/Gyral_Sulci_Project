import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import scipy.io as scio

basis_dir = '/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/temporal_basis'
basises = os.listdir(basis_dir)

for basis in basises:
    #signal = signal / max(abs(signal))
    print('basis comp is ', basis)
    sub_id = basis.split('_')[0]   #101107_gyri30.png
    part_id = basis.split('_')[1]
    comp_id = basis.split('_')[2].split('.')[0]
    print('sub id {}, part_id {}, comp_id {}'.format(sub_id, part_id, comp_id))
    gyri_temp = sub_id + '_gyri_' + comp_id +'_temporal_basis.txt'  #113316_gyri_1_temporal_basis.txt
    sulci_temp = sub_id + '_sulci_' + comp_id +'_temporal_basis.txt'

    gyri_temp = np.loadtxt(basis_dir+'/'+gyri_temp)
    sulci_temp = np.loadtxt(basis_dir+'/'+sulci_temp)
    
    plt.plot(gyri_temp, color = 'b', linewidth = 5 )
    plt.plot(sulci_temp, color = 'r', linewidth = 5 ) #'#ffb07c'
    print('pcc is ',np.corrcoef(gyri_temp, sulci_temp)[0,1])
    #plt.style.use('dark_background')
    #plt.axis('off')
    plt.show()
    print('end')







        
