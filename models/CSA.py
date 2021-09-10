#-*-coding:utf-8-*-

import torch
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torchvision import models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class CSA(BaseModel):
    def name(self):
        return 'CSAModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain
        
        self.lossNet = VGG16FeatureExtractor().cuda()
        
        self.ones_31 = torch.zeros(opt.batchSize, 1, 30, 30).fill_(1.0).type(torch.FloatTensor).cuda()
        self.zeros_31 = torch.zeros(opt.batchSize, 1, 30, 30).type(torch.FloatTensor).cuda()

        self.ones = torch.zeros( 256, 256).fill_(1.0).cpu().detach().numpy()

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        
        
        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)

        self.mask_global.zero_()
        
        
        self.mask_global[:, :, int(64)  : int(128) + int(64) ,int(64) : int(128) + int(64) ] = 1

        
        self.mask_type = opt.mask_type
        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()
            

        self.netG= networks.define_G(opt.input_nc_g, opt.ngf,
                                       opt, opt.init_type, self.gpu_ids, opt.init_gain)
        if self.isTrain:

            self.netD = networks.define_D(opt.input_nc, opt.ndf,opt.which_model_netD,opt.init_type, self.gpu_ids, opt.init_gain)
                     
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            
            self.L1 = torch.nn.L1Loss()   
            self.bce_loss = nn.BCELoss()           #L1损失

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            if self.isTrain:
                networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self,input,mask):

        input_A = input             #输入图片
        input_B = input.clone()     #复制输入图片
        
        
        input_mask=mask             #输入掩码

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        

        self.image_paths = 0

        if self.opt.mask_type == 'center':
            self.mask_global=self.mask_global      #使用定义好矩形的掩码

        elif self.opt.mask_type == 'random':
            self.mask_global.zero_()
            self.mask_global=input_mask        #使用输入的不规则掩码
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)

        
        self.input_A.narrow(1,0,1).masked_fill_(self.mask_global.bool(), 2*123.0/255.0 - 1.0)
        self.input_A.narrow(1,1,1).masked_fill_(self.mask_global.bool(), 2*104.0/255.0 - 1.0)
        self.input_A.narrow(1,2,1).masked_fill_(self.mask_global.bool(), 2*117.0/255.0 - 1.0)     #将输入图片打码


    def forward(self):
        self.real_A =self.input_A.to(self.device)   #被打码的真实图像
        
        self.image1,self.image2,self.image3,self.image4,self.m1,self.m2,self.m3= self.netG(self.real_A,self.mask_global)
        self.real_B = self.input_B.to(self.device)  # 真实图片


    def test(self):
        self.real_A =self.input_A.to(self.device)   #被打码的真实图像
        
        self.image1,self.image2,self.image3,self.image4,self.m1,self.m2,self.m3= self.netG(self.real_A,self.mask_global)
        self.real_B = self.input_B.to(self.device)      #真实图片

        # plt.show()

        
    def backward_D(self):
        
        fake_1 = self.image1   #得到最终生成图片
        fake_2 = self.image2   #得到最终生成图片
        fake_3 = self.image3   #得到最终生成图片
        fake_4 = self.image4   #得到最终生成图片
        # Real
        
        real_AB = self.real_B # GroundTruth
        
        mask_96 = self.m1
        mask_64 = self.m2
        mask_32 = self.m3
        
        self.pred_real_1,self.pred_real_2,self.pred_real_3,self.pred_real_4 = self.netD(real_AB*(1-mask_96),real_AB*(1-mask_64),real_AB*(1-mask_32),real_AB)
        real_loss_96 = self.bce_loss(self.pred_real_1, self.ones_31)
        real_loss_64 = self.bce_loss(self.pred_real_2, self.ones_31)
        real_loss_32 = self.bce_loss(self.pred_real_3, self.ones_31)
        real_loss_0 = self.bce_loss(self.pred_real_4, self.ones_31)
        
        self.pred_fake_1,self.pred_fake_2,self.pred_fake_3,self.pred_fake_4 = self.netD(fake_1.detach(),fake_2.detach(),fake_3.detach(),fake_4.detach())
        
        
        fake_loss_96 = self.bce_loss(self.pred_fake_1, self.zeros_31)
        fake_loss_64 = self.bce_loss(self.pred_fake_2, self.zeros_31)
        fake_loss_32 = self.bce_loss(self.pred_fake_3, self.zeros_31)
        fake_loss_0 = self.bce_loss(self.pred_fake_4, self.zeros_31)
        
        
        
        self.loss_D_fake = fake_loss_96+fake_loss_64+fake_loss_32+fake_loss_0
        self.loss_D_real = real_loss_96+real_loss_64+real_loss_32+real_loss_0
        self.loss_D = self.loss_D_fake + self.loss_D_real
      
        self.loss_D.backward()
        

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_1 = self.image1   #得到最终生成图片
        fake_2 = self.image2   #得到最终生成图片
        fake_3 = self.image3   #得到最终生成图片
        fake_4 = self.image4   #得到最终生成图片

        real_AB = self.real_B # GroundTruth

        mask = self.mask_global.float().cuda()
        mask_96 = self.m1
        mask_64 = self.m2
        mask_32 = self.m3
        
        real_B_feats = self.lossNet(real_AB)
        fake_B_feats = self.lossNet(fake_4)
        
        style_loss = self.style_loss(real_B_feats, fake_B_feats)
        preceptual_loss = self.preceptual_loss(real_B_feats, fake_B_feats)

        tv_loss = networks.cal_tv(fake_1)+networks.cal_tv(fake_2)+networks.cal_tv(fake_3)+networks.cal_tv(fake_4)
        
        
        
        p_fake_1,p_fake_2,p_fake_3,p_fake_4 = self.netD(fake_1,fake_2,fake_3,fake_4)
        
        self.loss_G_GAN = self.bce_loss(p_fake_1, self.zeros_31)+self.bce_loss(p_fake_2, self.zeros_31)+self.bce_loss(p_fake_3, self.zeros_31)+self.bce_loss(p_fake_4, self.zeros_31)
        self.loss_G_GAN = -self.loss_G_GAN 
        self.loss_G_L1 = 0
        self.loss_G_L1 += (self.L1(self.image1, self.real_B*(1-mask_96))+self.L1(self.image2, self.real_B*(1-mask_64))+self.L1(self.image3, self.real_B*(1-mask_32))+self.L1(self.image4, self.real_B)) * self.opt.lambda_A
        hole_loss = torch.mean(torch.abs((fake_4 - real_AB) * mask))
        
        
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN* self.opt.gan_weight 
        
        self.loss_G += tv_loss * self.opt.tv_weight + hole_loss * self.opt.hole_weight + preceptual_loss*self.opt.preceptual_weight+style_loss*self.opt.style_weight
                        
                        
        self.loss_G.backward()
        

 
    def style_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value
    
    
    
    def preceptual_loss(self, A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value
    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        


    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D', self.loss_D.data.item())
                            ])

    def get_current_visuals(self):

        real_A =self.real_A.data
        fake_B = self.image4.data

        real_B =self.real_B.data

        

        return real_A,real_B,fake_B


    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.gpu_ids)
        
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        
    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)
        


