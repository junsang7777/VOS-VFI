# Video Object Segmentation-aware Video Frame Interpolation (ICCV 2023)

This is official pytorch implementation of the paper "[Video Object Segmentation-aware Video Frame Interpolation](https://openaccess.thecvf.com/content/ICCV2023/html/Yoo_Video_Object_Segmentation-aware_Video_Frame_Interpolation_ICCV_2023_paper.html)" in ICCV 2023



## Abstract

Video frame interpolation (VFI) is a very active research topic due to its broad applicability to many applications, including video enhancement, video encoding, and slow-motion effects. VFI methods have been advanced by improving the overall image quality for challenging sequences containing occlusions, large motion, and dynamic texture. This mainstream research direction neglects that foreground and background regions have different importance in perceptual image quality. Moreover, accurate synthesis of moving objects can be of utmost importance in computer vision applications. In this paper, we propose a video object segmentation (VOS)-aware training framework called VOS-VFI that allows VFI models to interpolate frames with more precise object boundaries. Specifically, we exploit VOS as an auxiliary task to help train VFI models by providing additional loss functions, including segmentation loss and bi-directional consistency loss. From extensive experiments, we demonstrate that VOS-VFI can boost the performance of existing VFI models by rendering clear object boundaries. Moreover, VOS-VFI displays its effectiveness on multiple benchmarks for different applications, including video object segmentation, object pose estimation, and visual tracking.

## Enviornments
+ PyTorch
+ CUDA 11+
+ cupy-cuda
+ python 3.8
+ torchvision
+ pyiqa (for metrics)


## Train

### Prepare training data
1. Download Vimeo90k trainind data from [vimeo triplet dataset](http://toflow.csail.mit.edu/).
2. Using Vimeo90k dataset & precompute_mask.py, generate Object Segmentation Mask.

### Begin to train
1. Run train.py with following command.
      python train.py --train [dir_to_vimeo_triplet] --out_dir [dir_to_output_folder]
3. You might have to change many other options (epochs, learning rate, hyper parameters, etc.)

## Test

### Evaluation
1. The Evaluation part is same as existing VFI models.
2. For evaluation, you need the checkpoint file.
3. Run evaluation.py with following command.
   ```
      python evaluation.py --out_dir [output_dir] --checkpoint [checkpoint_dir] --config [configuration_dir]
   ```


### Video Interpolation
1. To interpolate and evaluate video datasets, Run interpolate_video_folder+accuracy_example.py
2. This example is the test code on video object segmentation dataset (DAVIS2016) for x2 odd frames.

### Two-frame interpolation 
1. To interpolate a frame between arbitary two fraems you have, run interpolte_twoframe.py with following command.
   ```
   python interpolate_twoframe.py --first_frame [first_frame] --second_frame [second_frame] --output_frame [output_frame] --checkpoint [checkpoint_dir] --config [configuration_dir]
   ```


## Citation
      
        @InProceedings{Yoo_2023_ICCV,
          author    = {Yoo, Jun-Sang and Lee, Hongjae and Jung, Seung-Won},
          title     = {Video Object Segmentation-aware Video Frame Interpolation},
          booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
          month     = {October},
          year      = {2023},
          pages     = {12322-12333}
        }
      
## Acknowledgements
This code is based on [HyeongminLee/AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch)
