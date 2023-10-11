import os
import time
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

semantic_weight = DeepLabV3_ResNet101_Weights.DEFAULT
semantic_model = deeplabv3_resnet101(weights=semantic_weight)
semantic_model.eval()
preproces = semantic_weight.transforms(resize_size=None)

img_root = '/ssd1/works/downloads/VFI_DATA/vimeo_triplet/sequences'
dst_flow_root = '/ssd1/works/downloads/VFI_DATA/vimeo_triplet/masks'
if not os.path.exists(dst_flow_root):
    os.mkdir(dst_flow_root)
for idx, name in enumerate(sorted(os.listdir(img_root))):
    img_dir = os.path.join(img_root, name)
    flo_dir = os.path.join(dst_flow_root, name)
    if not os.path.exists(flo_dir):
        os.mkdir(flo_dir)
    st_time = time.time()
    for subidx, subname in enumerate(sorted(os.listdir(img_dir))):

        print('%d / %d, %d / %d, remain' % (subidx, len(sorted(os.listdir(img_dir))), idx, len(sorted(os.listdir(img_root)))))
        img_subdir = os.path.join(img_dir, subname)
        flo_subdir = os.path.join(flo_dir, subname)
        img_path1 = os.path.join(img_subdir, 'im1.png')
        img_path2 = os.path.join(img_subdir, 'im2.png')
        img_path3 = os.path.join(img_subdir, 'im3.png')

        if not os.path.exists(flo_subdir):
            os.mkdir(flo_subdir)

        img1 = read_image(img_path1)
        img2 = read_image(img_path2)
        img3 = read_image(img_path3)

        batch1 = preproces(img1).unsqueeze(0)
        batch2 = preproces(img2).unsqueeze(0)
        batch3 = preproces(img3).unsqueeze(0)

        prediction1 = semantic_model(batch1)
        prediction1 = prediction1['out']
        prediction2 = semantic_model(batch2)
        prediction2 = prediction2['out']
        prediction3 = semantic_model(batch3)
        prediction3 = prediction3['out']

        normalized_masks1 = prediction1.softmax(dim=1)
        normalized_masks2 = prediction2.softmax(dim=1)
        normalized_masks3 = prediction3.softmax(dim=1)

        mask1 = torch.argmax(normalized_masks1[0],dim=0).numpy()
        mask2 = torch.argmax(normalized_masks2[0],dim=0).numpy()
        mask3 = torch.argmax(normalized_masks3[0],dim=0).numpy()

        clip1 = mask1.copy()
        clip1[clip1>0] = 1
        clip2 = mask2.copy()
        clip2[clip2>0] = 1
        clip3 = mask3.copy()
        clip3[clip3>0] = 1

        plt.imsave(flo_subdir+'/mask1.png',clip1,cmap='gray')
        plt.imsave(flo_subdir+'/mask2.png',clip2,cmap='gray')
        plt.imsave(flo_subdir+'/mask3.png',clip3,cmap='gray')

