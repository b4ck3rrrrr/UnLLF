import torch.nn as nn
import torch
from torch.nn.functional import grid_sample,adaptive_avg_pool2d
from .loss import Lossv3,Lossv9,Lossv5,get_smooth_loss,noOPAL,Lossv4,get_bv_loss,inverse_photometric_loss,compute_noise,get_distillation_loss
from .context_adjustment_layer import *
from .basemodel import BaseModel, init_net, get_optimizer
from .layers import *
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import os
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

        
        
        self.teacher_model = TeacherModel(self.device)
        save_dir = os.path.join('./checkpoints', 'OPENet', '2025-04-04-14-50')
        load_filename = '%s_net_%s.pth' % ('204', 'EPI')  # 教师模型的MSE：2.257
        load_path = os.path.join(save_dir,load_filename)
        self.teacher_model.load_state_dict(torch.load(load_path, map_location=str(self.device)))  # 加载权重文件
        self.teacher_model.to(self.device)
        
        
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
        isTrain = self.isTrain

        if isTrain:
            with torch.no_grad():  # 用于推理阶段，禁用梯度计算，减少显存占用并提高计算效率。
                self.teacher_outputs = self.teacher_model(self.input)

        
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
        self.loss_distillation = get_distillation_loss(self.output, self.teacher_outputs, self.epoch,0.5)
        # 蒸馏损失比光度一致性损失小一个数量级
        # print('loss_total',self.loss_total)
        # print('loss_distillation',self.loss_distillation)
        self.loss_total += self.loss_distillation 
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



class DSConv(nn.Module):
    """Depth-wise Separable Conv (3×3) + BN + ReLU"""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.depth = nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False)
        self.point = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn    = nn.BatchNorm2d(c_out)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.point(self.depth(x)))


class LightBlock(nn.Module):
    """Residual Bottleneck：DSConv(降维) → DSConv(升维)"""
    def __init__(self, c, ratio=4):
        super().__init__()
        hid = c // ratio
        self.conv = nn.Sequential(
            DSConv(c, hid),
            DSConv(hid, c)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv(x))


def make_layer(block, ch, num_blocks):
    layers = []
    for _ in range(num_blocks):
        layers.append(block(ch))
    return nn.Sequential(*layers)


# ------------------ 轻量 UNet ------------------
class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        # ----------- 编码器 -----------
        self.ch1   = DSConv(in_ch, base)
        self.conv1 = make_layer(LightBlock, base, 2)
        self.pool1 = nn.MaxPool2d(2)

        self.ch2   = DSConv(base, base*2)
        self.conv2 = make_layer(LightBlock, base*2, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.ch3   = DSConv(base*2, base*4)
        self.conv3 = make_layer(LightBlock, base*4, 2)
        self.pool3 = nn.MaxPool2d(2)

        self.ch4   = DSConv(base*4, base*8)
        self.conv4 = make_layer(LightBlock, base*8, 2)

        # ----------- 解码器 -----------
        self.up7   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.ch7   = DSConv(base*8, base*4)          # cat 后通道减半
        self.conv7 = make_layer(LightBlock, base*4, 2)

        self.up8   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.ch8   = DSConv(base*4, base*2)
        self.conv8 = make_layer(LightBlock, base*2, 2)

        self.up9   = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.ch9   = DSConv(base*2, base)
        self.conv9 = make_layer(LightBlock, base, 2)

        # ----------- 共享头 -----------
        self.reduce7 = nn.Conv2d(base*4, base, 1)
        self.reduce8 = nn.Conv2d(base*2, base, 1)
        self.head_d  = nn.Conv2d(base, 33, 1)   # 视差
        self.head_c  = nn.Conv2d(base, 33, 1)   # 置信度（可选）

    def forward(self, x):
        # ----- encoder -----
        c1 = self.conv1(self.ch1(x))
        c2 = self.conv2(self.ch2(self.pool1(c1)))
        c3 = self.conv3(self.ch3(self.pool2(c2)))
        c4 = self.conv4(self.ch4(self.pool3(c3)))

        # ----- decoder -----
        up7 = self.up7(c4)
        merge7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(self.ch7(merge7))

        up8 = self.up8(c7)
        merge8 = torch.cat([up8, c2], dim=1)
        c8 = self.conv8(self.ch8(merge8))

        up9 = self.up9(c8)
        merge9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(self.ch9(merge9))

        # ----- head -----
        disp9 = self.head_d(c9)
        # 如需多尺度输出：
        # disp7 = self.head_d(self.reduce7(c7))
        # disp8 = self.head_d(self.reduce8(c8))
        # return [disp7, disp8, disp9]
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
        
        
    def forward(self, x, gt):
        feats = []
        start = time.time()
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
#             occu = self.cal_occlusion(warpped_views)

#             disp_final, occu_final = self.finetune(out, occu, x[4])
#             disp_final = disp_final
#             mask.append(torch.where(disp_final<0.03,1,0).float())
#             disp.append(disp_final)
#             # occolusion.append(occu_final)
#         end = time.time()
#         t = end-start
#         print(1000*t)
#         mask = torch.where(torch.mean(torch.cat(mask, 1),1, True) < 1 ,0., 1.,) # all view==1, mask=1
#         disp = torch.cat(disp, 1)
#         disp_mean = torch.mean(disp, 1, True)
        
#         diff=[]
        
#         for i, x in enumerate(raw_warp_img[0]):

#             map = torch.abs(x - raw_warp_img[0][4])  # 将经过warp变换后的每张图与中心视图做差取绝对值，计算L1距离

#             map = torch.sum(map,dim=1)
#             mask = torch.where(map>0.5,1,0)

#             diff.append(mask)
#         mask_img = diff[1].permute(1, 2, 0).detach().cpu().numpy()
        



#         pil_image = Image.fromarray((mask_img[:, :, 0] * 255).astype(np.uint8))
        
#         # 保存为 JPEG 文件
#         output_path = 'mask.jpg'
#         pil_image.save(output_path)
        return out, raw_warp_img
    

    
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



class TeacherModel(nn.Module):
    def __init__(self,device):
        super().__init__()
        feats = 64
        self.device = device
        self.use_v = 9
        self.feat_extract = Feature(in_channels=3, out_channels=8)
        self.block3d = nn.ModuleList()
        self.fuse3d = nn.ModuleList()
        for _ in range(4):
            self.block3d.append(
                nn.Sequential(nn.Conv3d(8, 64, (self.use_v, 3, 3), 1, padding=(0, 1, 1)), nn.LeakyReLU(0.2, True)))
            self.fuse3d.append(nn.Sequential(nn.Conv2d(64, 32, 1, 1), nn.LeakyReLU(0.2, True)))
        feats *= 2
        self.fuse2d = nn.ModuleList()
        for j in range(4):
            self.fuse2d.append(nn.Sequential(nn.Conv2d(feats, feats, 3, 1, padding=1), nn.LeakyReLU(0.2, True),
                                             nn.Conv2d(feats, feats, 1, 1)))
            self.fuse2d.append(nn.LeakyReLU(0.2, True))
        self.fuse3 = nn.Conv2d(feats, 9, 3, 1, padding=1)
        self.relu3 = nn.Softmax(dim=1)
        self.finetune = ContextAdjustmentLayerv2()

    def forward(self, x):
        feats = []
        for xi in x:
            feat_i = self.feat_extract(xi)
            feats.append(feat_i)
        feats_angle = []
        for j in range(4):
            feats_tmp = torch.stack(feats[self.use_v * j:self.use_v * (j + 1)], dim=2)
            feats_tmp = self.block3d[j](feats_tmp)
            feats_tmp = torch.squeeze(feats_tmp, 2)
            feats_angle.append(self.fuse3d[j](feats_tmp))
        cv = torch.cat(feats_angle, 1)
        for j in range(4):
            cv = self.fuse2d[j * 2 + 1](self.fuse2d[j * 2](cv) + cv)
        prob = self.relu3(self.fuse3(cv))
        disp_raw = self.disparitygression(prob)
        disp_final = self.finetune(disp_raw, x[4])

        return disp_final

    def disparitygression(self, input):
        disparity_values = torch.linspace(-4, 4, 9, device=self.device)
        x = disparity_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(torch.multiply(input, x), 1)
        return out.unsqueeze(1)
    
if __name__ == "__main__":
    net = Unsup27_16_16()
    from thop import profile
    input = []
    for _ in range(36):
        input.append(torch.randn(1, 3, 48, 48))
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops/ 1e9))
