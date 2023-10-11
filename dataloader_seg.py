import os
import numpy as np
import glob
import cv2
import sys
import random
import math
import torch
from torch.utils.data import Dataset


### Segmentation mask +
class VimeoDataset_Seg(Dataset):
    def __init__(self, args):
        self.data_root = args.train
        self.phase = 'train'
        self.crop_size = args.patch_size

        self.load_data()

    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        self.meta_data = []
        #self.flow_data = []
        self.folder_data = []
        self.mask_data = []
        if self.phase == 'train':
            data_list = open(os.path.join(self.data_root, 'tri_trainlist.txt'), 'r')
        else:
            data_list = open(os.path.join(self.data_root, 'tri_testlist.txt'), 'r')

        for item in data_list:
            name = str(item).strip()
            if (len(name) <= 1):
                continue
            pair = sorted(glob.glob(os.path.join(self.data_root, 'sequences', name, '*.png')))
            #flow = sorted(glob.glob(os.path.join(self.data_root.replace('vimeo_triplet', ''), 'flows', name, '*.npy')))
            mask = sorted(glob.glob(os.path.join(self.data_root.replace('vimeo_triplet', ''), 'masks', name, '*.png')))
            self.meta_data.append(pair)
            #self.flow_data.append(flow)
            self.mask_data.append(mask)
            self.folder_data.append(name)

        self.nr_sample = len(self.meta_data)

    def aug(self, img0, gt, img1, h, w, mask0, mask_inter, mask1):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x + h, y:y + w, :]
        img1 = img1[x:x + h, y:y + w, :]
        gt = gt[x:x + h, y:y + w, :]
        mask0 = mask0[x:x + h, y:y + w, :]
        mask_inter = mask_inter[x:x + h, y:y + w, :]
        mask1 = mask1[x:x + h, y:y + w, :]
        return img0, gt, img1, mask0, mask_inter, mask1

    def getimg(self, index):
        data = self.meta_data[index]
        mask = self.mask_data[index]
        folder = self.folder_data[index]

        img0 = cv2.imread(data[0])
        gt = cv2.imread(data[1])
        img1 = cv2.imread(data[2])

        mask0 = cv2.imread(mask[0])
        mask_inter = cv2.imread(mask[1])
        mask1 = cv2.imread(mask[2])

        return img0, gt, img1, folder , mask0, mask_inter, mask1

    def __getitem__(self, index):
        img0, gt, img1, folder, mask0, mask_inter, mask1 = self.getimg(index)
        if self.phase == 'train':
            img0, gt, img1, mask0, mask_inter, mask1 = self.aug(img0, gt, img1, self.crop_size, self.crop_size, mask0, mask_inter, mask1)

            # # attention: can only be used without flow loss !!!
            # if random.uniform(0, 1) < 0.5:  # rotate
            #     img0 = np.ascontiguousarray(np.rot90(img0, k=1, axes=(0, 1)))
            #     img1 = np.ascontiguousarray(np.rot90(img1, k=1, axes=(0, 1)))
            #     gt = np.ascontiguousarray(np.rot90(gt, k=1, axes=(0, 1)))

            # if random.uniform(0, 1) < 0.5:  # color aug
            #     img0 = img0[:, :, ::-1]
            #     img1 = img1[:, :, ::-1]
            #     gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:  # vertical flip
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                mask0 = mask0[::-1]
                mask1 = mask1[::-1]
                mask_inter = mask_inter[::-1]
            if random.uniform(0, 1) < 0.5:  # horizontal flip
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                mask0 = mask0[:, ::-1]
                mask1 = mask1[:, ::-1]
                mask_inter = mask_inter[:, ::-1]
            if random.uniform(0, 1) < 0.5:  # reverse time
                tmp = img1
                img1 = img0
                img0 = tmp

                tmp_m = mask1
                mask1 = mask0
                mask0 = tmp_m
        else:
            h, w, _ = img0.shape
            # flow_gt = np.zeros((h, w, 4))

        if self.phase == 'train':
            img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
            gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
            img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

            mask0 = torch.from_numpy(mask0.astype('float32') / 255.).float().permute(2, 0, 1)
            mask_inter = torch.from_numpy(mask_inter.astype('float32') / 255.).float().permute(2, 0, 1)
            mask1 = torch.from_numpy(mask1.astype('float32') / 255.).float().permute(2, 0, 1)

            sample = {'img0': img0,
                      'img1': img1,
                      'gt': gt,
                      'folder': folder,
                      'mask0': mask0,
                      'mask1': mask1,
                      'mask_inter': mask_inter
                      }
        else:
            # pad HR to be mutiple of 64
            h, w, c = gt.shape
            if h % 64 != 0 or w % 64 != 0:
                h_new = math.ceil(h / 64) * 64
                w_new = math.ceil(w / 64) * 64
                pad_t = (h_new - h) // 2
                pad_d = (h_new - h) // 2 + (h_new - h) % 2
                pad_l = (w_new - w) // 2
                pad_r = (w_new - w) // 2 + (w_new - w) % 2
                img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT,
                                          value=0)  # cv2.BORDER_REFLECT
                img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            else:
                pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
            pad_nums = [pad_t, pad_d, pad_l, pad_r]

            #flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)
            img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
            gt = torch.from_numpy(gt).permute(2, 0, 1)
            # gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
            img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

            sample = {'img0': img0,
                      'img1': img1,
                      'gt': gt,
                      #'flow_gt': flow_gt,
                      'pad_nums': pad_nums,
                      'folder': folder}

        return sample