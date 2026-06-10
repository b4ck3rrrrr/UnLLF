import torch.nn as nn
import torch
from torch.nn.functional import grid_sample,adaptive_avg_pool2d
from .loss import Lossv3,Lossv9,Lossv5,get_smooth_loss,noOPAL,Lossv4,get_bv_loss,inverse_photometric_loss,compute_noise
from .context_adjustment_layer import *
from .basemodel import BaseModel, init_net, get_optimizer
from .layers import *
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.cm as cm
IDX = [[36+j for j in range(9)],
       [4+9*j for j in range(9)],
       [10*j for j in range(9)], 
       [8*(j+1) for j in range(9)]]


class OPENetmodel(BaseModel): # Basemodel
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1'] 
        if opt.losses.find('smooth') != -1:
            self.loss_names.append('smoothness')
        self.visual_names = ['center_input', 'output','label']
  
        self.model_names = ['EPI']
        net = eval(self.opt.net_version)(opt, self.device,self.isTrain)
        
        self.netEPI = init_net(net, opt.init_type, opt.init_gain, self.gpu_ids )                           
        self.use_v = opt.use_views

        self.center_index = self.use_v // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        self.lamda = opt.lamda

        self.test_loss_log = 0
        self.test_loss = torch.nn.L1Loss()
        if self.isTrain:
            # define loss functions
            self.criterionL1 = eval(self.opt.loss_version)(self.opt,self.device)
            self.optimizer = get_optimizer(opt, filter(lambda p: p.requires_grad, self.netEPI.parameters()), LR=opt.lr)
            self.optimizers.append(self.optimizer)
        self.study_mask = opt.study_mask
        if self.study_mask:
            if opt.use_dif_mask:
                self.mask = nn.Parameter(torch.zeros(2, 1, 64, 64, dtype=torch.float)).cuda()
            else:
                self.mask = 0
        else:
            self.mask = 0
    def set_input(self, inputs, epoch):
        self.supervise_view = inputs[0].to(self.device)
        
        
        self.epoch = epoch
        # self.supervise_view = rearrange(inputs[0].to(self.device), 'b c (h1 h) (w1 w) u v -> (b h1 w1) c h w u v', h1=8, w1=8)
        
        self.input = []
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, self.center_index,j])
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,self.center_index])    
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,j]) 
        for j in range(self.use_v):
            self.input.append(self.supervise_view[:,:,:,:, j,self.use_v-1-j])     
        
        self.input_7x7 = []
        for j in range(1,self.use_v-1):
            self.input_7x7.append(self.supervise_view[:,:,:,:, self.center_index,j])
        for j in range(1,self.use_v-1):
            self.input_7x7.append(self.supervise_view[:,:,:,:, j,self.center_index])    
        for j in range(1,self.use_v-1):
            self.input_7x7.append(self.supervise_view[:,:,:,:, j,j]) 
        for j in range(1,self.use_v-1):
            self.input_7x7.append(self.supervise_view[:,:,:,:, j,self.use_v-1-j]) 
        
        self.center_input = self.input[self.center_index]
        self.label = inputs[1].to(self.device)
    
    
    
    def forward(self):#############################################################################
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        
        self.output, self.raw_warp_img = self.netEPI(self.input,self.label)  # G(A)
        self.test_loss_log = 0 

    def backward_G(self):
        
        self.loss_L1 = self.criterionL1(self.output, self.input[:9],
                                        self.input[9:18])
        if self.raw_warp_img is None:
             self.loss_total = self.loss_L1
        else:
            self.loss_raw = 0
            for i, views in enumerate(self.raw_warp_img):
                if i==0:
                    self.loss_raw += self.criterionL1(self.output, self.input[:9])
                elif i==1:
                    self.loss_raw += self.criterionL1(self.output,
                                        self.input[9:18])
            
            self.loss_total = 0.6 * self.loss_L1 + 0.4 * self.loss_raw
            # self.loss_total = self.loss_L1
        self.loss_smoothness = get_smooth_loss(self.output, self.center_input,150) 
        self.loss_total += self.loss_smoothness
#         if 'smoothness' in self.loss_names and self.epoch > 2*self.opt.n_epochs:
#             self.loss_total += self.loss_smoothness 
        self.loss_total.backward()
        
    def optimize_parameters(self):
        self.netEPI.train()
        self.forward()                   
        self.optimizer.zero_grad()       
        self.backward_G()                   
        self.optimizer.step()           



class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(fn, fn, 3, padding=1)
        self.conv2 = nn.Conv2d(fn, fn, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(fn)
        self.norm2 = nn.BatchNorm2d(fn)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return identity + out        
        
        
def make_layer(block, p, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(p))
    return nn.Sequential(*layers)



class UNet(nn.Module):
    def __init__(self, in_ch):
        super(UNet, self).__init__()
        self.ch1 = nn.Conv2d(in_ch, 64, 3, 1, 1)
        self.conv1 = make_layer(ResidualBlock, 64, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.ch2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2 = make_layer(ResidualBlock, 128, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.ch3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3 = make_layer(ResidualBlock, 256, 2)
        self.pool3 = nn.MaxPool2d(2)

        self.ch4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4 = make_layer(ResidualBlock, 512, 2)


        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ch7 = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv7 = make_layer(ResidualBlock, 256, 2)
        self.head7_d = nn.Conv2d(256, 33, 1)
        self.head7_c = nn.Conv2d(256, 33, 1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ch8 = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv8 = make_layer(ResidualBlock, 128, 2)
        self.head8_d = nn.Conv2d(128, 33, 1)
        self.head8_c = nn.Conv2d(128, 33, 1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ch9 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv9 = make_layer(ResidualBlock, 64, 2)
        self.head9_d = nn.Conv2d(64, 33, 1)
        self.head9_c = nn.Conv2d(64, 33, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        c1 = self.conv1(self.relu(self.ch1(x)))
        
        p1 = self.pool1(c1)
        c2 = self.conv2(self.relu(self.ch2(p1)))
        p2 = self.pool2(c2)
        c3 = self.conv3(self.relu(self.ch3(p2)))
        p3 = self.pool3(c3)
        c4 = self.conv4(self.relu(self.ch4(p3)))
        # p4 = self.pool4(c4)
        # c5 = self.conv5(p4)
        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6 = self.conv6(merge6)
        
        # print(c4.shape)
        
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(self.relu(self.ch7(merge7)))
        disp7 = self.head7_d(c7)
        conf7 = self.head7_c(c7)
        out7 = [disp7, conf7]
        
        # print(c7.shape)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(self.relu(self.ch8(merge8)))
        disp8 = self.head8_d(c8)
        conf8 = self.head8_c(c8)
        out8 = [disp8, conf8]

        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(self.relu(self.ch9(merge9)))

        disp9 = self.head9_d(c9)
        conf9 = self.head9_c(c9)
        out9 = [disp9, conf9]

        return disp9

class StageBlock(nn.Module):
    def __init__(self, opt):
        super(StageBlock, self).__init__()




        self.recon_init=torch.nn.ConvTranspose2d(in_channels=33,out_channels=33,kernel_size=3,stride=1, padding=1,bias=False)
        self.proj=torch.nn.Conv2d(in_channels=33,out_channels=33,kernel_size=3,stride=1, padding=1,bias=False)
        self.recon=torch.nn.ConvTranspose2d(in_channels=33,out_channels=33,kernel_size=3,stride=1, padding=1,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.delta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda()) 
        self.eta=torch.nn.Parameter(torch.rand([1],dtype=torch.float32,requires_grad=True).cuda())
        self.refnet=UNet(in_ch=288)
        
    def forward(self, sub_lf):

        out_refnet = self.refnet(sub_lf)

#         err1 = self.recon(self.proj(out_refnet[0]) - out) #[bxy,c,u,v]
#         err2 = out - out_refnet[0] #[bxy,c,u,v]
#         disp_fliped_final = out - self.delta * (err1 + (1 - self.eta) * err2)

        # disp_fliped_final = out - self.delta * (err1 + self.eta * err2) #[bxy,c,u,v]
        
        return out_refnet
        # combined_input = torch.cat([disp_fliped.view(N*4, 1, h, w), lf[:,24].repeat(4,1,1,1)], dim=1)
        # 用ADMM
        # disp_fliped_final=self.refnet(combined_input)

        
def CascadeStages(block, opt):
    blocks = torch.nn.ModuleList([])
    for _ in range(opt.stageNum):
        blocks.append(block(opt))
    return blocks

        
        
        
        
        
        
# finetune 1
# views 36
# cycle 1
# transformer 0
# admm 1
class Unsup27_16_16(nn.Module):  # v3
    def __init__(self,opt,device, is_train=True):
        super().__init__() 
        self.is_train = is_train
        self.n_angle = 2
        feats = 64
        self.device = device
        self.use_v = opt.use_views
        self.grad_v = opt.grad_v
        self.feat_extract = Feature(in_channels=opt.input_c, out_channels=8)
        
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for _ in range(4):
            self.block3d.append(nn.Sequential(nn.Conv3d(8, 64, (self.use_v,3,3), 1, padding=(0, 1,1)),nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1),nn.LeakyReLU(0.2, True)))
        
        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1,padding=1),nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.finetune = ContextAdjustmentLayer()
        self.iterativeRecon = CascadeStages(StageBlock,opt)
        self.ch=torch.nn.Conv2d(in_channels=288,out_channels=33,kernel_size=3,stride=1, padding=1,bias=False)
        if opt.study_beta:
            if opt.use_dif_mask:
                self.beta = nn.Parameter(torch.tensor(opt.init_beta, dtype=torch.float32), requires_grad=True).cuda()
            else:
                self.beta = opt.init_beta
        else:
            self.beta = opt.init_beta
        
    def forward(self, x, gt):
        feats = []
        beta = self.beta
        # start = time.time()
        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)

        sub_lf = torch.stack(feats,dim=1) 

        
        n ,a ,c ,h , w =sub_lf.shape

        sub_lf = sub_lf.reshape(n, a*c, h, w)
        lf = sub_lf
        for stage in self.iterativeRecon:
            lf = stage(lf)

            
        out = lf[:, -1:, :, :] 

        occolusion = []
        disp = []
        mask = []
        raw_warp_img = []
        for j in range(self.n_angle):

            warpped_views = self.warp(gt, x[self.use_v*j:self.use_v*(j+1)], IDX[j])
            # raw_warp_img.append(warpped_views)

            raw_warp_img.append(warpped_views) 
            occu = self.cal_occlusion(warpped_views)

            disp_final, occu_final = self.finetune(out, occu, x[4])
            disp_final = disp_final*occu_final
            mask.append(torch.where(disp_final<0.03,1,0).float())
            disp.append(disp_final)
            # occolusion.append(occu_final)
        # end = time.time()
        # t = end-start
        # print(1000*t)
        mask = torch.where(torch.mean(torch.cat(mask, 1),1, True) < 1 ,0., 1.,) # all view==1, mask=1
        disp = torch.cat(disp, 1)
        disp_mean = torch.mean(disp, 1, True)
        
        diff=[]
        
        for i, x in enumerate(raw_warp_img[0]):

            map = torch.abs(x - raw_warp_img[0][4])  # 将经过warp变换后的每张图与中心视图做差取绝对值，计算L1距离

            map = torch.sum(map,dim=1)
            mask = torch.where(map>0.5,1,0)

            diff.append(mask)
        mask_img = diff[1].permute(1, 2, 0).detach().cpu().numpy()
        



        pil_image = Image.fromarray((mask_img[:, :, 0] * 255).astype(np.uint8))
        
        # 保存为 JPEG 文件
        output_path = 'mask.jpg'
        pil_image.save(output_path)
        return disp_mean, raw_warp_img
    

    
    def warp(self, disp, views_list, idx):
        B,C,H,W = views_list[0].shape
        x, y = torch.arange(0, H), torch.arange(0, W)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2
        disp = disp.squeeze(1)
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):
            u, v = divmod(idx[k], 9)
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))   
        return tmp

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4,4,9,device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)

    def cal_occlusion(self, views):
        views = torch.stack(views, -1) # B*C*H*W*9
        if self.grad_v == 9:
            grad = torch.mean(torch.abs(views[:,:,:,:,1:]- views[:,:,:,:,:8]), dim=[1,4])
        elif self.grad_v == 5:
            grad = torch.mean(torch.abs(views[:,:,:,:,3:7] - views[:,:,:,:,2:6]), dim=[1,4]) # B*H*W   
        return grad.unsqueeze(1)




