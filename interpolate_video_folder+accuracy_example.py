
import argparse
from PIL import Image
import torch
from torchvision import transforms
import models
import os
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable

import pyiqa

#from models.cdfi_adacof import CDFI_adacof
import skimage.metrics
#from lpips_pytorch import lpips
import numpy as np
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='Video Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='/ssd1/works/projects/AdaCoF-pytorch-segmentation/ada_ONLY_CORR/checkpoint/model_epoch045.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
parser.add_argument('--zpad', type=int, default=5, help='zero padding of frame name.')

parser.add_argument('--input_video', type=str, default='./test_video_davis')
parser.add_argument('--output_video', type=str, default='./CORR_45')

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



#args = parser.parse_args()
args = parser.parse_args(args=[])
torch.cuda.set_device(args.gpu_id)

# config_file = open(args.config, 'r')
# while True:
#     line = config_file.readline()
#     if not line:
#         break
#     if line.find(':') == 0:
#         continue
#     else:
#         tmp_list = line.split(': ')
#         if tmp_list[0] == 'kernel_size':
#             args.kernel_size = int(tmp_list[1])
#         if tmp_list[0] == 'flow_num':
#             args.flow_num = int(tmp_list[1])
#         if tmp_list[0] == 'dilation':
#             args.dilation = int(tmp_list[1])
# config_file.close()

model = models.Model(args)

print('Loading the model...')

checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
model.load(checkpoint['state_dict'])
NIQE = pyiqa.create_metric('niqe')
PI = pyiqa.create_metric('pi')
NIMA = pyiqa.create_metric('nima')
#%%

base_dir = args.input_video
print(base_dir)
import glob
import os
folders = sorted(glob.glob(base_dir+'/*'))
#folders

#%%
total_av_pi = 0
total_av_niqe = 0
total_av_nima = 0
total_av_lps = 0
total_av_ssim = 0
total_av_psnr = 0
total_frame_len = 0
for k in range(len(folders)):
    frame_len = len([name for name in os.listdir(folders[k]) if os.path.isfile(os.path.join(folders[k], name))])
    print(frame_len)
    folder_name = folders[k].split('/')[-1]
    if not os.path.exists(args.output_video+'/'+folder_name):
        os.makedirs(args.output_video+'/'+folder_name)

    half_len = int(frame_len / 2)
    if frame_len % 2 == 0 :
        xxx = int(frame_len/2) - 1
    else :
        xxx = int(frame_len/2)
    folder_av_lps = 0
    folder_av_ssim = 0
    folder_av_psnr = 0
    folder_av_nima = 0
    folder_av_niqe = 0
    folder_av_pi = 0
    interpol_frame_len = 0
    for idx in range(xxx):
        idx += args.index_from
        print(idx, '/', xxx, end='\r')

        frame_name1 = folders[k] + '/' + str(2*idx).zfill(args.zpad) + '.jpg'
        frame_name2 = folders[k] + '/' + str(2*(idx + 1)).zfill(args.zpad) + '.jpg'
        frame_gt = folders[k] + '/' + str(2*(idx)+1).zfill(args.zpad) + '.jpg'

        frame1 = to_variable(transform(Image.open(frame_name1)).unsqueeze(0))
        frame2 = to_variable(transform(Image.open(frame_name2)).unsqueeze(0))

        model.eval()
        #print(frame1.shape)
        #if frame1.shape[-1] > 900 :
        #    frame_out = model(F.interpolate(frame1,scale_factor=(1/2)), F.interpolate(frame2,scale_factor=1/2))
        #    frame_out = F.interpolate(frame_out,scale_factor=2)
        #else:
        frame_out = model(frame1, frame2)
        ########### PSNR etc..
        gt = transform(Image.open(frame_gt))
        gt_tensor = gt.unsqueeze(0)

        #lps = lpips(gt_tensor[0], frame_out.to('cpu').detach(), net_type='squeeze')
        lps=0
        #print(frame_out.shape)
        #print(gt_tensor.shape)
        psnr = skimage.metrics.peak_signal_noise_ratio(gt.numpy(),frame_out[0].to('cpu').detach().numpy(),data_range=1)
        ssim = skimage.metrics.structural_similarity(np.transpose(gt.numpy(),(1,2,0)), np.transpose(frame_out[0].to('cpu').detach().numpy(),(1,2,0)),channel_axis=2)
        folder_av_psnr += psnr
        folder_av_ssim += ssim
        folder_av_lps += lps
        interpol_frame_len +=1
        ###################
        # interpolate
        imwrite(frame1.clone(), args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2 + args.index_from).zfill(args.zpad) + '.jpg', range=(0, 1))
        imwrite(frame_out.clone(), args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.jpg', range=(0, 1))
        #niqe = NIQE(args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.jpg')
        #pi = PI(args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.jpg')
        #nima = NIMA(args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.jpg')
        #folder_av_pi += pi.item()
        #folder_av_niqe += niqe.item()
        #folder_av_nima += nima.item()
        #print(args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2+ args.index_from).zfill(args.zpad) + '.jpg')
        #print(args.output_video + '/' + folder_name + '/' + str((idx - args.index_from) * 2 + 1 + args.index_from).zfill(args.zpad) + '.jpg')
    print('Folder : %s, Folder_PSNR : %.6f,  Folder_SSIM : %.6f,  Folder_LPIPS : %.6f'%(folder_name, folder_av_psnr/interpol_frame_len,folder_av_ssim/interpol_frame_len,folder_av_lps/interpol_frame_len))
    print('Folder : %s, folder_av_niqe : %.6f,  folder_av_pi : %.6f,  folder_av_nima : %.6f'%(folder_name, folder_av_niqe/interpol_frame_len,folder_av_pi/interpol_frame_len,folder_av_nima/interpol_frame_len))
    # last frame
    print(xxx, '/', xxx)
    frame_name_last = folders[k] + '/' + str(2*(xxx + args.index_from)).zfill(args.zpad) + '.jpg'
    frame_last = to_variable(transform(Image.open(frame_name_last)).unsqueeze(0))
    imwrite(frame_last.clone(), args.output_video + '/' + folder_name + '/' + str((xxx) * 2 + args.index_from).zfill(args.zpad) + '.jpg', range=(0, 1))
    #print(frame_name_last)
    if frame_len%2 == 0:
        _frame_name_last = folders[k] + '/' + str((frame_len + args.index_from - 1)).zfill(args.zpad) + '.jpg'
        #print(_frame_name_last)
        _frame_last = to_variable(transform(Image.open(_frame_name_last)).unsqueeze(0))
        imwrite(_frame_last.clone(), args.output_video + '/' + folder_name + '/' + str((frame_len - 1) + args.index_from).zfill(args.zpad) + '.jpg', range=(0, 1))
    total_av_psnr += folder_av_psnr
    total_av_ssim += folder_av_ssim
    total_av_lps += folder_av_lps
    total_frame_len += interpol_frame_len
    #total_av_niqe += folder_av_niqe
    #total_av_pi += folder_av_pi
    #total_av_nima += folder_av_nima
print('Total_PSNR : %.6f,  Total_SSIM : %.6f,  Total_LPIPS : %.6f'%(total_av_psnr/total_frame_len,total_av_ssim/total_frame_len,total_av_lps/total_frame_len))
#print('total_av_niqe : %.6f,  total_av_pi : %.6f,  total_av_nima : %.6f'%(total_av_niqe/total_frame_len,total_av_pi/total_frame_len,total_av_nima/total_frame_len))
