
import argparse
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cuda = True if torch.cuda.is_available() else False
#cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda' if cuda else 'cpu')

if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0005, help='adam: learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--model_saved_path', type=str, default='/media/shawey/SSD8T/GyraSulci_Motor/Twin_Transformers/models_saved')

parser.add_argument('--corr_threshold', type=float, default=0.2, help='voxel signal correlates with stimulus')
parser.add_argument('--depth', type=int, default=6, help='depth of ViT')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--output_fea_dim', type=int, default=1024, help='dimension of ViT output')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--voxels_num', type=int, default=60000) #59412 -> 60000
parser.add_argument('--time_length', type=int, default=284)   #284 -> 284
parser.add_argument('--time_unit', type=int, default=284) #because time_length is only 284, take 284 as time_unit 
parser.add_argument('--datafolder', type=str, default='/media/shawey/SSD8T/GyraSulci_Motor/DataNpy')
parser.add_argument('--spa_comm', type=int, default=20)   # number of common features
parser.add_argument('--temp_comm', type=int, default=20)   # number of common features
parser.add_argument('--spa_temp_comm', type=int, default=0)   # number of common features
parser.add_argument('--total_comm', type=int, default=100)   # number of features
parser.add_argument('--mask_path', type=str, default='/home/xiaow/IJCAI2022/HCP3T4mm')   # number of common features
opt = parser.parse_args()
