#use autoencoder(transformer) for matrix decompostion
#t*n = t*k * k*n
from sklearn.utils import shuffle
from vit import *
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import *           
import nibabel as nib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import h5py
import time

#284 seconds, 
#output dim of weight is componest num * voxels_num
weight_net = weight_ViT( signal_length = opt.voxels_num,   #input 284 * 28600
                patch_len = 600,   # 284 * 60000 divide into 284 * 600, (60000 / 600 = 100) patches
                patch_time = opt.time_length,   
                # in HCP 7T moive, the time_length is too long, that why patch time is 284 
                n_components = opt.total_comm,    
                # num_patches = 28600/path_len ; 100 * 286 = 28600; weight dim 100 * 28600   
                depth = 6, 
                heads = 8, 
                dim = 1024,
                dim_head = 256,    # dim_head * heads = dim
                mlp_dim = 2048, 
                pool = 'cls', 
                dropout = 0.1, 
                emb_dropout = 0.1)

# output dim of inde_com_net is 284 * 100
atom_net = atom_ViT( signal_length = opt.voxels_num,   #input 284*28600 as smallest unit
                patch_len = int(opt.voxels_num ),   # 284 * 28600 divide into 1 * 28600, 284 / 1 = 284 patches
                n_components =  opt.total_comm,   #   284 * 100 as atom matrix
                time_unit = opt.time_unit, #284 s           
                dim= 1024,     
                depth = 6, 
                heads = 8, 
                dim_head = 256,    # dim_head * heads = dim
                mlp_dim = 2048, 
                pool = 'cls', 
                dropout = 0.1, 
                emb_dropout = 0.1)

siameseNet = SiameseNet(atom_net, weight_net)

atom_net = nn.DataParallel(atom_net)
atom_net.to(device)
weight_net = nn.DataParallel(weight_net)
weight_net.to(device)
siameseNet.to(device)

if os.path.exists(( opt.model_saved_path + '/weight_net.pkl')):
    weight_net.load_state_dict(torch.load( opt.model_saved_path + '/weight_net.pkl') )
if os.path.exists(( opt.model_saved_path + '/atom_net.pkl')):
    atom_net.load_state_dict(torch.load( opt.model_saved_path + '/atom_net.pkl'))

seed = 42
seed_everything(seed)

# loss function
mae_criterion = nn.L1Loss()
# optimizer
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,itertools.chain(atom_net.parameters(),weight_net.parameters(),\
                                  )), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay =0)
# scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=opt.gamma, patience=50000, )

subs = os.listdir(opt.datafolder)
reco_subs =[]
for sub in subs:
    sub_id = sub.split('_')[0]
    if sub_id not in reco_subs:
        reco_subs.append(sub_id)
loss_coffs = [10, 10, 1, 1]  #spatial loss + recon loss + temporal loss + temporal norm loss

max_mask_len = opt.voxels_num   

all_subs = reco_subs
tr_subs = reco_subs[0:535]
te_subs = reco_subs[535:]
for epoch in range(opt.epochs):
    comb_cnt = 0
    #random.shuffle(tr_subs)
    for sub in tr_subs:
        sub_id = sub.split('_')[0]
        print('sub id ', sub_id)
        sulci = opt.datafolder + '/' + sub_id + "_MOTOR_sulci.npy" #889579_MOTOR_sulci.npy
        gyri = opt.datafolder + '/' + sub_id + '_MOTOR_gyri.npy' #889579_MOTOR_gyri.npy
        if( not ( os.path.exists(sulci) and os.path.exists(gyri) ) ):
            continue
        if(sub_id not in reco_subs):
            reco_subs.append(sub_id)
        start_t = time.time()
        print('current lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        comb_cnt += 1
        print('epoch {} comb_cnt {} '.format( epoch, comb_cnt))
        epoch_tra_loss = 0    # each comb reports a mse
        epoch_val_loss = 0

        fmri_data = np.load(sulci)
        if(max_mask_len > fmri_data.shape[1]):
            add_array = np.zeros((opt.time_length, max_mask_len - fmri_data.shape[1]))
            fmri_data1 = np.concatenate((fmri_data, add_array), axis=1)
        else:
            fmri_data1 = fmri_data[:,0:max_mask_len]
        print('sulci max {}, min {}'.format(fmri_data1.max(), fmri_data1.min()))
        #print(len(fmri_data1[[fmri_data1!=0]])/284)
        #------------------------------------------------------

        fmri_data = np.load(gyri)
        if(max_mask_len > fmri_data.shape[1]):
            add_array = np.zeros((opt.time_length, max_mask_len - fmri_data.shape[1]))
            fmri_data2 = np.concatenate((fmri_data, add_array), axis=1)
        else:
            fmri_data2 = fmri_data[:,0:max_mask_len]
        print('gyri max {}, min {}'.format(fmri_data2.max(), fmri_data2.min()))
        #print(len(fmri_data2[[fmri_data2!=0]])/284)
        
        fmri_data1=torch.from_numpy(fmri_data1.reshape(1,opt.time_unit, opt.voxels_num)).float().to(device)
        fmri_data2=torch.from_numpy(fmri_data2.reshape(1,opt.time_unit, opt.voxels_num)).float().to(device)

        dic1, dic2,inde_comps1,inde_comps2 = siameseNet(fmri_data1,fmri_data2) #inde_comps1,inde_comps2

        spatial_loss = comm_spa_loss( inde_comps1[:,0:opt.spa_comm,:],\
                    inde_comps2[:,0:opt.spa_comm,:] ) #spatial features MAE
        #for dictionary, column; for spatial components, row 
        reco_signal_loss = mae_criterion(torch.matmul(dic1, inde_comps1)[fmri_data1!=0],fmri_data1[fmri_data1!=0]) + \
            mae_criterion(torch.matmul(dic2, inde_comps2)[fmri_data2!=0],fmri_data2[fmri_data2!=0] )
        temporal_loss = temporal_corr_score(dic1[:,:,opt.spa_comm:(opt.temp_comm+opt.spa_comm)], \
            dic2[:,:,opt.spa_comm:(opt.temp_comm+opt.spa_comm)]) 
        
        temporal_norm = (temporal_basis_norm(dic1) +temporal_basis_norm(dic2) )/2
        temporal_norm_loss = torch.max( torch.tensor(0).to(device), \
                                 temporal_norm - 1.0) 

        print('tra: temporal_loss {},  reco_signal_loss {}' \
                     .format(temporal_loss.detach(), reco_signal_loss.detach()))
        print('spatial_loss {}, temporal_norm_loss {}'.format(spatial_loss.detach(), temporal_norm_loss.detach()  )  )
        total_loss = loss_coffs[0]* spatial_loss + loss_coffs[1]* reco_signal_loss + \
                         loss_coffs[2]* temporal_loss + loss_coffs[3]* temporal_norm_loss 
            
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)
        epoch_tra_loss += total_loss.detach() 

            #############################################################################
    for sub in te_subs:
        with torch.no_grad():
            siameseNet.eval()
            atom_net.eval()
            weight_net.eval()

            sub_id = sub.split('_')[0]
            print('sub id ', sub_id)
            sulci = opt.datafolder + '/' + sub_id + "_MOTOR_sulci.npy" #889579_MOTOR_sulci.npy
            gyri = opt.datafolder + '/' + sub_id + '_MOTOR_gyri.npy' #889579_MOTOR_gyri.npy
            if( not ( os.path.exists(sulci) and os.path.exists(gyri) ) ):
                continue
            fmri_data = np.load(sulci)
            if(max_mask_len > fmri_data.shape[1]):
                add_array = np.zeros((opt.time_length, max_mask_len - fmri_data.shape[1]))
                fmri_data1 = np.concatenate((fmri_data, add_array), axis=1)
            else:
                fmri_data1 = fmri_data[:,0:max_mask_len]
            print('sulci max {}, min {}'.format(fmri_data1.max(), fmri_data1.min()))
            #------------------------------------------------------

            fmri_data = np.load(gyri)
            if(max_mask_len > fmri_data.shape[1]):
                add_array = np.zeros((opt.time_length, max_mask_len - fmri_data.shape[1]))
                fmri_data2 = np.concatenate((fmri_data, add_array), axis=1)
            else:
                fmri_data2 = fmri_data[:,0:max_mask_len]
            print('gyri max {}, min {}'.format(fmri_data2.max(), fmri_data2.min()))

            fmri_data1=torch.from_numpy(fmri_data1.reshape(1,opt.time_unit, opt.voxels_num)).float().to(device)
            fmri_data2=torch.from_numpy(fmri_data2.reshape(1,opt.time_unit, opt.voxels_num)).float().to(device)

            dic1, dic2,inde_comps1,inde_comps2  = siameseNet(fmri_data1,fmri_data2) #inde_comps1,inde_comps2

            spatial_loss = comm_spa_loss( inde_comps1[:,0:opt.spa_comm,:],\
                    inde_comps2[:,0:opt.spa_comm,:] ) #spatial features MAE
            #for dictionary, column; for spatial components, row 
            reco_signal_loss = mae_criterion(torch.matmul(dic1, inde_comps1)[fmri_data1!=0],fmri_data1[fmri_data1!=0]) + \
                mae_criterion(torch.matmul(dic2, inde_comps2)[fmri_data2!=0],fmri_data2[fmri_data2!=0] )
            temporal_loss = temporal_corr_score(dic1[:,:,opt.spa_comm:(opt.temp_comm+opt.spa_comm)], \
                dic2[:,:,opt.spa_comm:(opt.temp_comm+opt.spa_comm)]) 
            
            temporal_norm = (temporal_basis_norm(dic1) +temporal_basis_norm(dic2) )/2
            temporal_norm_loss = torch.max( torch.tensor(0).to(device), \
                                    temporal_norm - 1.0) 

            print('eval: temporal_loss {},  reco_signal_loss {}' \
                        .format(temporal_loss.detach(), reco_signal_loss.detach()))
            print('spatial_loss {}, temporal_norm_loss {}'.format(spatial_loss.detach(), temporal_norm_loss.detach()  )  )
            total_loss = loss_coffs[0]* spatial_loss + loss_coffs[1]* reco_signal_loss + \
                            loss_coffs[2]* temporal_loss + loss_coffs[3]* temporal_norm_loss 

            epoch_val_loss = total_loss.detach() 

            #save predicted diams to files
            temporal_comps1 = dic1[:,:,0:opt.total_comm].squeeze()
            temporal_comps2 = dic2[:,:,0:opt.total_comm].squeeze()
            _,num_comps = temporal_comps1.shape
            if (epoch >= 0):
                    for i in range(num_comps):
                        temporal_basis = np.array(temporal_comps1[:,i].cpu())
                        np.savetxt('./temporal_basis/'+str(sub_id)+'_sulci_'+str(i)+'_temporal_basis.txt', temporal_basis)
                        #plt.cla()
                        #plt.plot(temporal_basis)
                        #plt.show()
                        #plt.savefig('./figs/'+str(sub1)+'_'+str(i)+'_temporal.png')
                    for i in range(num_comps):
                        temporal_basis = np.array(temporal_comps2[:,i].cpu())
                        np.savetxt('./temporal_basis/'+str(sub_id)+'_gyri_'+str(i)+'_temporal_basis.txt', temporal_basis)
                        #plt.cla()
                        #plt.plot(temporal_basis)
                        #plt.show()
                        #plt.savefig('./figs/'+str(sub1)+'_'+str(i)+'_temporal.png')

            inde_comps1 = inde_comps1[:,0:opt.total_comm,:].squeeze()
            inde_comps2 = inde_comps2[:,0:opt.total_comm,:].squeeze()
            num_comps,_ = inde_comps1.shape
            if (epoch >= 0):
                torch.save(inde_comps1,"./figs/"+ str(sub_id) +"_sulci_weight.pkl")
                torch.save(inde_comps2,"./figs/"+ str(sub_id) +"_gyri_weight.pkl")
                pass

            print(
                f"comb : {epoch+1} - tra_loss: {epoch_tra_loss:.4f} - val_loss: {epoch_val_loss:.4f}\n"
            )
        
    torch.save(atom_net.state_dict(), opt.model_saved_path + '/atom_net.pkl')
    torch.save(weight_net.state_dict(), opt.model_saved_path + '/weight_net.pkl')

    end_t = time.time()
    print('1 comb time usage ', end_t - start_t)
    print('subs len', len(reco_subs))
            
  



