#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import Parameter
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler
from .ContextualAttention import *
from .partialconv2d import PartialConv2d
from rn import RN_B, RN_L

###############################################################################
# Functions
###############################################################################

def cal_tv(image):
    temp = image.clone()
    temp[:,:,:256-1,:] = image[:,:,1:,:]
    re = ((image-temp)**2).mean()
    temp = image.clone()
    temp[:,:,:,:256-1] = image[:,:,:,1:]
    re += ((image-temp)**2).mean()
    return re

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net.cuda()
    
    
def define_G(input_nc, ngf, opt,init_type='normal',  gpu_ids=[], init_gain=0.02):
    

    netG = Generator(input_nc, ngf, opt)
    
    print('[CREATED] MODEL')     

    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    
    
    if which_model_netD == 'n_layers':
        netD = Discriminator(input_nc, ndf)


    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d  M' % (num_params/1e6))

################################### ***************************  #####################################
###################################  Progressiveiy Inpainting Image Based on a Forked-Then-Fused Decoder Network  #####################################
################################### ***************************  ####################################


def conv_down(dim_in, dim_out):
    return nn.Sequential(
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(dim_out)
    )


def conv_up(dim_in, dim_out):
    return nn.Sequential(
        nn.ReLU(True),
        nn.ConvTranspose2d(dim_in, dim_out,kernel_size=3, stride=1,padding=1, bias=False),
        nn.BatchNorm2d(dim_out)
    )


class PConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, act='ReLU', use_norm=True):
        super(PConvLayer, self).__init__()
        self.conv = PartialConv2d(in_channels, out_channels, 
                        kernel_size, stride, padding, bias = False)
        # self.norm = nn.InstanceNorm2d(out_channels, track_running_stats=False)
        self.norm = RN_B(out_channels)
        self.use_norm = use_norm
        if act == 'ReLU':
            self.act = nn.ReLU(True)
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU(0.2, True)
        elif act == 'Tanh':
            self.act = nn.Tanh()

    def forward(self, x, mask):
        x, mask_update = self.conv(x, mask)
        if self.use_norm:
            x = self.norm(x, mask)
        x = self.act(x)
        return x, mask_update

class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, ngf, opt):
        super(CoarseGenerator, self).__init__()

        self.conv1 = PConvLayer(3, ngf, kernel_size=3, stride=2, padding=1, act='LeakyReLU')  #  down 输出为128 128 64
        self.conv2 = PConvLayer(ngf, ngf*2, kernel_size=3, stride=2, padding=1, act='LeakyReLU')  # down 输出为64 64 128
        self.conv3 = PConvLayer(ngf, ngf*2, kernel_size=3, stride=1, padding=1, act='LeakyReLU')  # down 输出为64 64 128

        self.conv4 = PConvLayer(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1)  # down 输出为32 32 256
        self.conv5 = PConvLayer(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256
        self.conv6 = PConvLayer(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256
        self.conv7 = PConvLayer(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')# up scale_factor=2

        self.upconv8_up = conv_up(ngf * 4 * 2, ngf * 4)  # up 输出为32 32 256
        self.upconv9_up = conv_up(ngf * 4 * 2, ngf * 4)  # up 输出为32 32 256
        self.upconv10_up = conv_up(ngf * 4 * 2, ngf * 4)  # up 输出为32 32 256
        self.upconv11_up = conv_up(ngf * 4 * 2, ngf * 4)  # up 输出为32 32 256
        self.upconv12_up = conv_up(ngf * 4 * 2, ngf * 2)  # up 输出为64 64 128
        self.upconv13_up = conv_up(ngf * 4 * 2, ngf * 2)  # up 输出为64 64 128

        self.upconv8_bottom = PConvLayer(ngf * 4 *2 , ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256
        self.upconv9_bottom = PConvLayer(ngf * 4 * 2, ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256
        self.upconv10_bottom = PConvLayer(ngf * 4 * 2, ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256
        self.upconv11_bottom = PConvLayer(ngf * 4 * 2, ngf * 4, kernel_size=3, stride=1, padding=1)  # down 输出为32 32 256
        self.upconv12_bottom = PConvLayer(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=1, padding=1)  # down 输出为64 64 128
        self.upconv13_bottom = PConvLayer(ngf * 4 * 2, ngf * 2, kernel_size=3, stride=1, padding=1)  # down 输出为64 64 128

        if opt.CA_type == "single":
            self.CA_0 = ContextualAttentionModule(patch_size=3, propagate_size=3)
            self.CA_1 = ContextualAttentionModule(patch_size=3, propagate_size=3, fuse=False)
            print("single CA")

        elif opt.CA_type == "parallel":

            self.CA_0 = ParallelContextualAttention(256)
            self.CA_1 = ParallelContextualAttention(512, fuse=False)
            print("parallel CA")

        self.SqueezeExc = SEModule(128 * 2)
        self.combiner = nn.Conv2d(128 * 2, 128, kernel_size=1)


        self.upconv14 = conv_up(ngf * 2 * 2, ngf)  # up 输出为128 128 32
        self.upconv15 = conv_up(ngf * 2, 3)  # up 输出为256 256 3




class Generator(nn.Module):
    def __init__(self,input_dim, ngf,opt):
        super(Generator, self).__init__()
        
        self.CoarseGenerator_1 = CoarseGenerator(input_dim, ngf,opt)
        self.CoarseGenerator_2 = CoarseGenerator(6, ngf, opt )
        self.CoarseGenerator_3 = CoarseGenerator(6, ngf, opt)
        self.CoarseGenerator_4 = CoarseGenerator(6, ngf, opt)
        
       
    def forward(self, x , mask ):
		
        mask = mask.float().cuda()
        out_1, m1= self.CoarseGenerator_1(torch.cat((x, (1 - mask).expand(x.size(0), 1, x.size(2), x.size(3)).type_as(x)), dim=1),mask,x)
        
        out_2,m2 = self.CoarseGenerator_2(torch.cat((x, out_1), 1), m1,out_1)
     
        out_3,m3= self.CoarseGenerator_3(torch.cat((x, out_2), 1),m2, out_2)

        out_4,m4= self.CoarseGenerator_4(torch.cat((x, out_3), 1),m3,out_3)
        

        return out_1, out_2, out_3, out_4, m1, m2, m3
        
        
        
class SEModule(nn.Module):
    def __init__(self, num_channel, squeeze_ratio = 1.0):
        super(SEModule, self).__init__()
        self.sequeeze_mod = nn.AdaptiveAvgPool2d(1)
        self.num_channel = num_channel
        
        blocks = [ nn.Linear(num_channel, int(num_channel*squeeze_ratio)),
                   nn.ReLU(),
                   nn.Linear(int(num_channel*squeeze_ratio), num_channel),
                   nn.Sigmoid()]
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        ori = x
        x = self.sequeeze_mod(x)
        x = x.view(x.size(0), 1, self.num_channel)
        x = self.blocks(x)
        x = x.view(x.size(0), self.num_channel, 1, 1)
        x = ori * x
        return x
        
        
        
def conv_stage(dim_in, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, 4, 2, 1,bias=False),
        nn.LeakyReLU(0.2,True),
        nn.BatchNorm2d(dim_out)
    )

        
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(NLayerDiscriminator, self).__init__()
        self.conv0 = conv_stage(input_nc, ndf)
        self.conv1 = conv_stage(ndf, ndf * 2)
        self.conv2 = conv_stage(ndf * 2, ndf * 4)
        self.conv3 = nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)
        self.conv4 = nn.Conv2d(ndf * 8, 1, 4, 1, 1)
        
       

    def forward(self, input):
        conv0_out = self.conv0(input) 
        conv1_out = self.conv1(conv0_out) 
        conv2_out = self.conv2(conv1_out) 
        conv3_out = self.conv3(conv2_out)  
        out = self.conv4(conv3_out)  
        out = torch.sigmoid(out)
        return out
        

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__()
        self.discriminator1=NLayerDiscriminator(input_nc, ndf)
        self.discriminator2=NLayerDiscriminator(input_nc, ndf)
        self.discriminator3=NLayerDiscriminator(input_nc, ndf)
        self.discriminator4=NLayerDiscriminator(input_nc, ndf)
        
        
    def forward(self, input1,input2, input3,input4):
        pred_1=self.discriminator1(input1)
        pred_2=self.discriminator2(input2)
        pred_3=self.discriminator3(input3)
        pred_4=self.discriminator4(input4)
        

        return pred_1,pred_2,pred_3,pred_4

