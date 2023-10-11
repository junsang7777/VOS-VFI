import os
import torch
import utility
from utility import to_variable
import torch.nn as nn
from inference_core import InferenceCore
import matplotlib.pyplot as plt
import numpy as np

MEAN = torch.FloatTensor([0.485, 0.456, 0.406]).view(1 ,3 ,1 ,1).cuda()
STD = torch.FloatTensor([0.229, 0.224, 0.225]).view(1 ,3 ,1 ,1).cuda()

class Trainer:
    def __init__(self, args, train_loader, test_loader, my_model, my_loss, seg_model, start_epoch=0):
        self.args = args
        self.train_loader = train_loader
        self.max_step = self.train_loader.__len__()
        self.test_loader = test_loader
        self.model = my_model
        self.loss = my_loss
        self.current_epoch = start_epoch
        self.seg_model = seg_model
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        self.segmentation_loss = True
        #self.softmap_loss = True
        self.criterion_seg = nn.BCELoss().cuda()

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        self.result_dir = args.out_dir + '/result'
        self.ckpt_dir = args.out_dir + '/checkpoint'

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.logfile = open(args.out_dir + '/log.txt', 'w')

        # Initial Test
        self.model.eval()
        self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')

    def train(self):
        # Train
        self.model.train()
        for batch_idx, sample in enumerate(self.train_loader):
            img0 = to_variable(sample['img0'])
            img1 = to_variable(sample['img1'])
            gt = to_variable(sample['gt'])

            self.optimizer.zero_grad()

            output = self.model(img0, img1)
            loss = self.loss(output, gt, [img0, img1])

            if self.segmentation_loss == True:
                mask0 = sample['mask0']
                mask1 = sample['mask1']
                mask_inter = sample['mask_inter']
                rgb0 = (img0 - MEAN) / STD
                rgb1 = (img1 - MEAN) / STD
                rgb_gt = (gt - MEAN) / STD
                rgb_out = (output['frame1'] - MEAN) / STD

                Forward_rgb = torch.cat((rgb0.unsqueeze(1), rgb_out.unsqueeze(1)), dim=1)
                Backward_rgb = torch.cat((rgb1.unsqueeze(1), rgb_out.unsqueeze(1)), dim=1)

                Forward_mask = torch.cat((mask0.unsqueeze(1), mask_inter.unsqueeze(1)), dim=1)
                Forward_mask = Forward_mask[:, :, 0].unsqueeze(2)
                Backward_mask = torch.cat((mask1.unsqueeze(1), mask_inter.unsqueeze(1)), dim=1)
                Backward_mask = Backward_mask[:, :, 0].unsqueeze(2)

                processor = InferenceCore(self.seg_model, Forward_rgb, 1, top_k=20, mem_every=5)
                processor.interact(Forward_mask[:, 0], 0, Forward_rgb.shape[1])

                out_masks = torch.zeros((processor.t, 1, *Forward_rgb.shape[-2:]), dtype=torch.float32,
                                        device='cuda:0')
                for ti in range(processor.t):
                    prob = processor.prob[:, ti]
                    # print(prob.shape)
                    if processor.pad[2] + processor.pad[3] > 0:
                        prob = prob[:, :, processor.pad[2]:-processor.pad[3], :]
                    if processor.pad[0] + processor.pad[1] > 0:
                        prob = prob[:, :, :, processor.pad[0]:-processor.pad[1]]

                    out_masks[ti] = torch.argmax(prob, dim=0)
                forward_out_masks = out_masks[:, 0][1]

                #######################################################################################
                processor = InferenceCore(self.seg_model, Backward_rgb, 1, top_k=20, mem_every=5)
                processor.interact(Backward_mask[:, 0], 0, Backward_rgb.shape[1])

                out_masks = torch.zeros((processor.t, 1, *Backward_rgb.shape[-2:]), dtype=torch.float32,
                                        device='cuda:0')
                for ti in range(processor.t):
                    prob = processor.prob[:, ti]
                    # print(prob.shape)
                    if processor.pad[2] + processor.pad[3] > 0:
                        prob = prob[:, :, processor.pad[2]:-processor.pad[3], :]
                    if processor.pad[0] + processor.pad[1] > 0:
                        prob = prob[:, :, :, processor.pad[0]:-processor.pad[1]]

                    out_masks[ti] = torch.argmax(prob, dim=0)
                backward_out_masks = out_masks[:, 0][1]
                #######################################################################################


                # plt.imshow((out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)[1])
                # plt.show()
                #print(backward_out_masks.shape, forward_out_masks.shape, mask_inter[0][0].shape)
                seg_consistent_loss = self.criterion_seg(backward_out_masks, forward_out_masks)
                seg_forward_loss = self.criterion_seg(forward_out_masks, mask_inter[0][0].cuda())
                seg_backward_loss = self.criterion_seg(backward_out_masks, mask_inter[0][0].cuda())

                seg_loss = seg_forward_loss + seg_backward_loss + seg_consistent_loss
            #    if seg_loss > 8 :
            #        seg_loss = 0
                seg_loss = seg_loss * 1
                loss += seg_loss

            #if self.softmap_loss == True:
            #    output_softmap = self.model(mask0.cuda(), mask1.cuda())
            #    loss_softmap = self.loss(output_softmap, mask_inter.cuda(), [mask0, mask1])
            #    loss += loss_softmap

            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}{:<12s}{:<20.16f} '
                      .format('Train Epoch: ', '[' + str(self.current_epoch) + '/' + str(self.args.epochs) + ']',
                              'Step: ', '[' + str(batch_idx) + '/' + str(self.max_step) + ']',
                              'train loss: ', loss.item(), 'seg loss', seg_loss))
        self.current_epoch += 1
        self.scheduler.step()

    def test(self):
        # Test
        torch.save({'epoch': self.current_epoch, 'state_dict': self.model.get_state_dict()}, self.ckpt_dir + '/model_epoch' + str(self.current_epoch).zfill(3) + '.pth')
        self.model.eval()
        self.test_loader.Test(self.model, self.result_dir, self.current_epoch, self.logfile, str(self.current_epoch).zfill(3) + '.png')
        self.logfile.write('\n')

    def terminate(self):
        return self.current_epoch >= self.args.epochs

    def close(self):
        self.logfile.close()
