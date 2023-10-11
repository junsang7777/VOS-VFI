
from math import log10, sqrt
import cv2
import numpy as np
import glob
import os

folders = sorted(glob.glob('/ssd1/works/projects/AdaCoF-pytorch/test_video_davis'+'/*'))
for k in range(len(folders)):
        frame_len = len([name for name in os.listdir(folders[k]) if os.path.isfile(os.path.join(folders[k], name))])
        print(frame_len)
        folder_name = folders[k].split('/')[-1]
        if frame_len % 2 == 0 :
            xxx = int(frame_len/2) - 1
        else :
            xxx = int(frame_len/2)

        for idx in range(xxx):
            frame_gt = folders[k] + '/' + str(2*(idx)+1).zfill(5) + '.jpg'
            break

#%%

gt = cv2.imread('/ssd1/works/downloads/VOS_DATA/DAVIS/JPEGImages/480p/drift-straight/00033.jpg')
gt_mask = cv2.imread('/ssd1/works/downloads/VOS_DATA/DAVIS/Annotations/480p/drift-straight/00033.png',cv2.IMREAD_GRAYSCALE)
taget = cv2.imread('/ssd1/works/projects/AdaCoF-pytorch/ada_seg_50/drift-straight/00033.jpg')
gt_mask = (gt_mask>0)
stack_gt_mask = np.stack((gt_mask,gt_mask,gt_mask), axis=2)
stack_gt_mask.shape

#%%

import matplotlib.pyplot as plt
plt.imshow(gt_mask)

#%%

masked_taget = taget * stack_gt_mask
masked_gt = gt * stack_gt_mask
plt.imshow(masked_gt)
plt.show()
plt.imshow(masked_taget)

#%%

np.max(masked_gt)

#%%

count_true = sum(gt_mask.reshape(-1))
count_true

#%%

def PSNR(original, compressed, count_true):
    #mse = np.mean((original - compressed) ** 2)
    mse = np.sum((original - compressed) ** 2) / count_true
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

PSNR(masked_gt,masked_taget,count_true)

#%%


folders = sorted(glob.glob('/ssd1/works/projects/AdaCoF-pytorch/test_video_davis'+'/*'))
annotation_dir = '/ssd1/works/downloads/VOS_DATA/DAVIS/Annotations/480p/'
target_dir = '/ssd1/works/projects/EMA-VFI_LOAD/EMA-VFI-OURS/'
Total_PSNR = 0
for k in range(len(folders)):
        frame_len = len([name for name in os.listdir(folders[k]) if os.path.isfile(os.path.join(folders[k], name))])
        folder_name = folders[k].split('/')[-1]
        print(folder_name, frame_len)
        if frame_len % 2 == 0 :
            xxx = int(frame_len/2) - 1
        else :
            xxx = int(frame_len/2)
        folder_PSNR = 0
        for idx in range(xxx):
            frame_gt = folders[k] + '/' + str(2*(idx)+1).zfill(5) + '.jpg'
            frame_gt_mask = annotation_dir + folder_name + '/' + str(2*idx+1).zfill(5) + '.png'
            frame_target = target_dir + folder_name + '/' + str(2*idx+1).zfill(5) + '.jpg'

            gt = cv2.imread(frame_gt)
            img = cv2.imread(frame_target)
            mask = cv2.imread(frame_gt_mask,cv2.IMREAD_GRAYSCALE)
            mask = (mask>0)
            stack_gt_mask = np.stack((mask,mask,mask), axis=2)
            count_true = sum(mask.reshape(-1))
            masked_taget = img * stack_gt_mask
            masked_gt = gt * stack_gt_mask
            psnr = PSNR(masked_gt,masked_taget,count_true)
            #print(psnr)
            folder_PSNR += psnr
        print('Name : %s  ,  PSNR  :  %f' %(folder_name, folder_PSNR/xxx))
        Total_PSNR += folder_PSNR / xxx
print('Total : ', Total_PSNR / len(folders) )
