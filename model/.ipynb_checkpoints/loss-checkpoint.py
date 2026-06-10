import torch
import torch.nn as nn
from torch.nn.functional import grid_sample,pad,softmax


IDX = [[36+j for j in range(9)],
       [4+9*j for j in range(9)],
       [10*j for j in range(9)], 
       [8*(j+1) for j in range(9)]]


class Lossv9(nn.Module):
    def __init__(self, opt, device):
        super(Lossv9, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size + 2 * self.pad), torch.arange(0, self.patch_size + 2 * self.pad)
        # 第一个参数是起始值，第二个参数是结束值（不包括），而步长默认为 1。
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)  # 1*H*W*2
        '''
        meshgrid函数会生成一个H行W列的二维坐标，从左上角（0，0）开始一直到右下角（H-1,W-1），将行坐标和列坐标分别保存，用两个张量表示
>>> import torch
>>> h = 6
>>> w = 10
>>> xs,ys = torch.meshgrid(torch.arange(h), torch.arange(w))
>>> xs.shape
torch.Size([6, 10])
>>> ys.shape
torch.Size([6, 10])
>>> xs
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])
>>> ys
tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
>>> xys = torch.stack([xs, ys], dim=-1)
>>> xys.shape
torch.Size([6, 10, 2])
        '''
        # .to(device) 将张量移动到指定的设备（例如 GPU）上进行计算。
        ones = torch.ones((opt.batch_size, 1, opt.input_size, opt.input_size)).to(device)
        zeros = torch.zeros((opt.batch_size, 1, opt.input_size, opt.input_size)).to(device)
        '''
        这里的9个mask对应的便是论文中的一维遮挡模式，1维度表示的不是通道而是角坐标，即某个方向中的某个视图的索引
        '''
        mask_1 = ones.repeat(1, 9, 1, 1)  # 分别在 batch 角坐标 H W 四个维度上重复 1、9、1、1 次，形成新的张量，所有的视图全部保留，对应论文中的j=0
        mask_2 = torch.cat((zeros.repeat(1, 1, 1, 1), ones.repeat(1, 8, 1, 1)), 1)  # 将第一个视图遮挡掉，保留后八个视图，j=1
        mask_3 = torch.cat((zeros.repeat(1, 2, 1, 1), ones.repeat(1, 7, 1, 1)), 1)
        mask_4 = torch.cat((zeros.repeat(1, 3, 1, 1), ones.repeat(1, 6, 1, 1)), 1)
        mask_5 = torch.cat((zeros.repeat(1, 4, 1, 1), ones.repeat(1, 5, 1, 1)), 1)  # 将前四个视图遮挡掉，保留后五个视图，j=7
        mask_6 = torch.cat((ones.repeat(1, 8, 1, 1), zeros.repeat(1, 1, 1, 1)), 1)
        mask_7 = torch.cat((ones.repeat(1, 7, 1, 1), zeros.repeat(1, 2, 1, 1)), 1)
        mask_8 = torch.cat((ones.repeat(1, 6, 1, 1), zeros.repeat(1, 3, 1, 1)), 1)
        mask_9 = torch.cat((ones.repeat(1, 5, 1, 1), zeros.repeat(1, 4, 1, 1)), 1)  # 保留前五个视图，将后四个视图遮挡掉，j=8

        # mask_up = torch.cat((ones.repeat(1,1,1,1),zeros.repeat(1,8,1,1)),1)
        # mask_down = torch.cat((zeros.repeat(1,8,1,1),ones.repeat(1,1,1,1)),1)

        self.mask = [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8,
                     mask_9]  # self.mask 是一个包含了 9 个不同的掩码张量的列表
        # self.mask = [mask_1,mask_3,mask_5,mask_7,mask_9,mask_up,mask_down]

    def forward(self, pred_disp, *args):
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):  # raw\\ loss
            total_loss += self.cal_l1(pred_disp)    # 检查变量 pred_disp 是否是列表类型。如果 pred_disp 是列表类型，则表示有多个视差预测值需要计算损失。
        else:  # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss

    # 这个函数的功能是基于视差（disp）对一组视图列表（views_list）（4个方向中的其中一个方向）进行 warping 操作。
    def warpping(self, disp, views_list, idx):

        disp = disp.squeeze(1)  # 将通道维度去除 B*1*H*W → B*H*W
        B, C, H, W = views_list[0].shape  # 根据输入的 views_list 中第一个视图的形状，获取了 B、C、H、W 四个维度的值。
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        # assert 语句通常用于在开发和调试阶段检查代码的一致性和正确性
        # 如果这个断言失败（即 H 不等于 self.patch_size 或者 W 不等于 self.patch_size），就会触发 AssertionError，并输出 "size is different!" 的错误信息。
        tmp = []
        # .to(disp) 将复制后的网格坐标张量转换到与 disp 张量相同的设备上，以确保二者在同一设备上进行计算。
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp)  # B*H*W*2
        for k in range(
                9):  # ————————        5       7          ——————————
            u, v = divmod(idx[k], 9)  # 计算u  v 坐标   ######## k+2  k+1  分别表示对于5*5和7*7视图索引的开始位置
            # 计算一个归一化的网格坐标 grid，根据grid中的坐标顺序对input进行重新采样，从而生成新的output。
            '''
            此处生成的grid便是经过warp变换后的采样坐标，根据grid采样之后便是经过warp变换后的视图
            有个问题是此处的行坐标和列坐标为什么对调了？？？为什么用的是减法而不是论文里面的加法？？？
            '''
            grid = torch.stack([  # 首先需要根据H和W的大小，对输入特征图元素坐标位置进行规范化。
                torch.clip(meshgrid[:, :, :, 1] - disp * (v - 4), 0, W - 1),
                torch.clip(meshgrid[:, :, :, 0] - disp * (u - 4), 0, H - 1)
            ], -1) / (W - 1) * 2 - 1  # B*H*W*2  归一化到-1，1
            # gird的作用在于提供一批用于在输入特征图上进行元素采样的位置坐标。grid的元素值通常在[-1,1]之间，(-1,-1) 表示取输入特征图左上角的元素，(1,1) 表示取输入特征图右下角的元素。
            # 输出特征图中的元素值是从根据grid所提供的位置坐标在输入特征图中采样得到的。
            # 当grid设定为对输入特征图元素坐标位置规范化结果的水平翻转形式时，也就实现了对输入特征图的水平翻转。
            # grid里面的数据表示的是坐标信息，根据grid怎么采样都可以（可以采样输入特征图的任意位置、输出特征图的大小是任意的）
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
            # 如果grid中的某个坐标直接对应于输入特征图元素位置的规范化结果中的某个坐标，则直接把对应的值取过来就行。
            # 但如果grid中的某个坐标不能直接对应于输入特征图元素位置的规范化结果中的所有坐标，则需要根据不同的插值方式（比如bilinear）在输入特征图中进行插值。
            # 至于align_corners这个参数，一般和插值方式mode搭配使用，表示在插值时像素的对齐方式，有两种选择，分别是True和False。
            # 如果把一个像素点看做一个正方形的话，True表示坐标位于对应正方形的中心。False表示坐标位于对应正方形的角。
        return tmp

    def cal_l1(self, x_list):
        maps = []
        for x in x_list:
            map = torch.abs(x - x_list[4])  # 将经过warp变换后的每张图与中心视图做差取绝对值，计算L1距离
            map = map[:, 0, :, :] * 0.299 + map[:, 1, :, :] * 0.587 + map[:, 2, :, :] * 0.114  # 进行灰度图转换
            maps.append(map)

        # maps B*9*H*W
        maps = torch.stack(maps, dim=1)  # 将列表maps在1维度上堆叠形成张量maps,此处的维度1代表的不再是通道，而是视图列表中的9个视图和中心视图的L1距离
        distance_list = []
        '''
        选取不同的遮挡模式将对应的视图遮挡掉，计算所有像素所有遮挡的代价，选择每个像素代价最小的遮挡模式
        不过代码里面每个方向的最佳遮挡是分开计算的，而论文里面看起来似乎是整个二维角坐标同时计算的
        '''
        for i in range(len(self.mask)):  #
            distance = self.mask[i] * maps  # 将每个mask与maps进行点乘
            distance = torch.sum(distance, dim=1) / torch.sum(self.mask[i], dim=1)  # 在维度1上求和然后规范化，对应论文中的公式5
            distance_list.append(distance)

        diff = torch.abs(distance_list[8] - distance_list[4])   # 对应论文中的公式6，随便选取两个遮挡模式的代价做差
        diff = torch.where(diff < 0.01, 1, 0)  # B*H*W 根据条件对张量 diff 中的元素进行替换，将小于 0.01 的元素替换为 1，大于等于 0.01 的元素替换为 0。

        distances = torch.stack(distance_list, dim=1)  # 将列表distance_list在1维度上进行堆叠形成维度为B*9*H*W的张量distances
        distances, _ = torch.min(distances, dim=1)  # 沿1维度进行最小代价的选择。(B*H*W)返回两个张量，一个是由1维度上最小代价构成的张量 distances，另一个包含对应的索引信息（在这里用下划线 _ 表示）。
        # 下划线 _ 用于表示一个临时的、不需要的值。表明我们暂时不需要这个值，只对第一个返回值感兴趣。
        distances = distances * (1 - diff) + distance_list[0] * diff    # 如果小于阈值0.01则选择全一的遮挡模式，否则选择代价最小的遮挡模式，对应论文中的公式6
        # 在 PyTorch 中，如果 diff 是一个张量，那么 1 - diff 将执行逐元素的减法操作，即将 diff 中每个元素都从 1 中减去。
        # 这是一种广播（broadcasting）机制，其中标量值 1 被自动扩展为与 diff 张量相同的形状，然后执行逐元素的减法运算。
        loss = torch.mean(distances)  # 对 distances 张量中的所有元素进行求和，然后除以元素的总数，得到这些元素的平均值，即均值损失值。

        return loss



class Lossv5(nn.Module):
    def __init__(self, opt,device):
        super(Lossv5, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x,y), -1).unsqueeze(0) # 1*H*W*2

        ones = torch.ones((opt.batch_size,1,opt.input_size,opt.input_size)).to(device)
        zeros = torch.zeros((opt.batch_size,1,opt.input_size,opt.input_size)).to(device)

        mask_1 = ones.repeat(1,9,1,1)
        # mask_2 = torch.cat((zeros.repeat(1,1,1,1),ones.repeat(1,8,1,1)),1)
        mask_3 = torch.cat((zeros.repeat(1,2,1,1),ones.repeat(1,7,1,1)),1)
        # mask_4 = torch.cat((zeros.repeat(1,3,1,1),ones.repeat(1,6,1,1)),1)
        mask_5 = torch.cat((zeros.repeat(1,4,1,1),ones.repeat(1,5,1,1)),1)
        # mask_6 = torch.cat((ones.repeat(1,8,1,1),zeros.repeat(1,1,1,1)),1)
        mask_7 = torch.cat((ones.repeat(1,7,1,1),zeros.repeat(1,2,1,1)),1)
        # mask_8 = torch.cat((ones.repeat(1,6,1,1),zeros.repeat(1,3,1,1)),1)
        mask_9 = torch.cat((ones.repeat(1,5,1,1),zeros.repeat(1,4,1,1)),1)


        # self.mask = [mask_1,mask_2,mask_3,mask_4,mask_5,mask_6,mask_7,mask_8,mask_9]
        self.mask = [mask_1,mask_3,mask_5,mask_7,mask_9]
        # self.mask = [mask_1,mask_3,mask_5,mask_7,mask_9,mask_up,mask_down]

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):
        maps = []
        for x in x_list:
            map = torch.abs(x-x_list[4])
            map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
            maps.append(map)

        # maps B*9*H*W
        maps = torch.stack(maps,dim=1)
        distance_list = []

        for i in range(len(self.mask)):
            distance = self.mask[i]*maps
            distance = torch.sum(distance,dim=1)/torch.sum(self.mask[i],dim=1)
            distance_list.append(distance)

        diff = torch.abs(distance_list[4]-distance_list[2])
        diff = torch.where(diff<0.01,1,0)

        distances = torch.stack(distance_list,dim=1)
        distances, _ = torch.min(distances,dim=1)
        distances = distances*(1-diff)+distance_list[0]*diff
        loss = torch.mean(distances)

        return loss

class Lossv3(nn.Module):
    def __init__(self, opt,de):
        super(Lossv3, self).__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
  
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):
        map_up = torch.abs(x_list[0]-x_list[4])
        map_up = map_up[:,0,:,:]*0.299+map_up[:,1,:,:]*0.587+map_up[:,2,:,:]*0.114
        map_down = torch.abs(x_list[8]-x_list[4])
        map_down = map_down[:,0,:,:]*0.299+map_down[:,1,:,:]*0.587+map_down[:,2,:,:]*0.114

        mask_up = torch.unsqueeze(torch.where(map_up-map_down<self.alpha,1,0),dim=1)
        mask_down = torch.unsqueeze(torch.where(map_down-map_up<self.alpha,1,0),dim=1)

        loss = 0
        for j in range(4): ############## 5  7
            loss += self.base_loss(mask_up * x_list[j], mask_up * x_list[4]) ############ 2  3
        for j in range(5,9): ############## 5  7
            loss += self.base_loss(mask_down * x_list[j], mask_down * x_list[4]) ############ 2  3
        # loss = 0
        # for j in range(9): ############## 5   7
        #     loss += self.base_loss( x_list[j], x_list[4]) ############ 2  3
       
        return loss

class Lossv4(nn.Module):
    def __init__(self, opt,de):
        super().__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, epoch, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp,epoch)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]),epoch)
        return total_loss
      
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list, epoch):
        maps = []
        cv_gray = x_list[4][:,0,:,:]*0.299+x_list[4][:,1,:,:]*0.587+x_list[4][:,2,:,:]*0.114
        rate = 1/(cv_gray.detach()+2e-1)
        for i,x in enumerate(x_list):
            if i != 4:
                map = torch.abs(x-x_list[4])
                map = map[:,0,:,:]*0.299+map[:,1,:,:]*0.587+map[:,2,:,:]*0.114
                maps.append(map)
        maps = torch.stack(maps,dim=1)

        map_up = torch.mean(maps[:,0:4,:,:],dim=1)
        map_down = torch.mean(maps[:,4:,:,:],dim=1)
        if epoch>=30: map_circle = torch.mean(maps[:,2:6,:,:],dim=1)
        map_all = torch.mean(maps[:,:,:,:],dim=1)

        diff = torch.abs(map_up-map_down)
        diff = torch.where(diff<0.01,1,0)

        if epoch>=30: maps = torch.stack([map_up,map_down,map_circle,map_all],dim=1)
        else: maps = torch.stack([map_up,map_down,map_all],dim=1)

        map_min,_ = torch.min(maps,dim=1)
        loss = map_min*(1-diff)+map_all*diff
        loss = torch.mean(loss*rate)
       
        return loss

class noOPAL(nn.Module):
    def __init__(self, opt,de):
        super().__init__()
        self.base_loss = torch.nn.L1Loss()
        self.views = opt.use_views
        self.patch_size = opt.input_size
        self.center_index = self.views // 2
        self.alpha = opt.alpha
        self.pad = opt.pad
        x, y = torch.arange(0, self.patch_size+2*self.pad), torch.arange(0, self.patch_size+2*self.pad)
        self.meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0) # 1*H*W*2

    def forward(self, pred_disp,  *args): 
        '''
        pred_disp: (tensor) B*1*H*W
        views_x: (tensor list) [B*C*H*W]* 9
        views_y: (tensor list) [B*C*H*W]* 9
        views_45: (tensor list) [B*C*H*W]* 9
        views_135: (tensor list) [B*C*H*W]* 9
        '''
        total_loss = 0
        if isinstance(pred_disp, list):# raw loss
            total_loss += self.cal_l1(pred_disp)
        else: # final loss
            for i, views in enumerate(args):
                total_loss += self.cal_l1(self.warpping(pred_disp, views, IDX[i]))
        return total_loss
      
  
    def warpping(self, disp, views_list, idx):
        disp = disp.squeeze(1)
        B,C,H,W = views_list[0].shape
        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        tmp = []
        meshgrid = self.meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
        for k in range(9):  ##############   5       7                                                                                        #############
            u, v = divmod(idx[k], 9) ######## k+2  k+1
            grid = torch.stack([ 
                torch.clip(meshgrid[:,:,:,1]-disp*(v-4),0,W-1),
                torch.clip(meshgrid[:,:,:,0]-disp*(u-4),0,H-1)
            ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
            tmp.append(grid_sample(views_list[k], grid, align_corners=True))  ######## k+2  k+1
        return tmp

    def cal_l1(self, x_list):

        loss = 0
        for j in range(9): ############## 5   7
            loss += self.base_loss( x_list[j], x_list[4]) ############ 2  3
       
        return loss




def get_smooth_loss(disp, img, lamda):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-lamda*grad_img_x)
    grad_disp_y *= torch.exp(-lamda*grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()



def get_bv_loss(disp, img, sigma_color=0.1, sigma_spatial=1.0):
    """
    Computes the Bilateral Variation (BV) loss for a disparity image.
    The color image is used to guide the edge-aware weight computation.

    Args:
        disp (torch.Tensor): The predicted disparity map with shape (B, C, H, W).
        img (torch.Tensor): The input RGB image with shape (B, C, H, W).
        sigma_color (float): The standard deviation for color/brightness differences.
        sigma_spatial (float): The standard deviation for spatial differences.

    Returns:
        torch.Tensor: The BV loss.
    """
    # Compute disparity gradients
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    # Compute image gradients (guidance weight)
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

    # Compute bilateral weights
    weight_x = torch.exp(-grad_img_x * sigma_color)
    weight_y = torch.exp(-grad_img_y * sigma_color)

    # Apply weights to disparity gradients
    grad_disp_x *= weight_x
    grad_disp_y *= weight_y
    
    sigma_spatial_tensor = torch.tensor(sigma_spatial, dtype=torch.float32)  # 确保数据类型匹配
    bv_loss_x = torch.mean(grad_disp_x) * torch.exp(-1.0 / sigma_spatial_tensor)

    bv_loss_y = torch.mean(grad_disp_y) * torch.exp(-1.0 / sigma_spatial_tensor)

    return bv_loss_x + bv_loss_y


inverse_IDX =  [36,37,38,39,41,42,43,44,4,13,22,31,49,58,67,76,0,10,20,30,50,60,70,80,8,16,24,32,48,56,64,72]

def inverse_warp(disp_list,center):
    tmp = []
    x, y = torch.arange(0, 64), torch.arange(0, 64)
    meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)
    B,C,H,W = center.shape
    disp = disp_list[0]
    meshgrid = meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
    for i in range(32):
        disp = disp_list[i].squeeze(1)

        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        
                                                                                               #############
        u, v = divmod(inverse_IDX[i], 9) ######## k+2  k+1
        grid = torch.stack([ 
            torch.clip(meshgrid[:,:,:,1]-disp*(4-v),0,W-1),
            torch.clip(meshgrid[:,:,:,0]-disp*(4-u),0,H-1)
        ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
        tmp.append(grid_sample(center, grid, align_corners=True))  ######## k+2  k+1
    return tmp

def inverse_photometric_loss(disp_list,center,view_list):
    
    criterion = torch.nn.L1Loss()
    warp_view_list = inverse_warp(disp_list,center)
    visi_mask = []
    for i in range(16):
        cv_gray = warp_view_list[i][:,0,:,:]*0.299+warp_view_list[i][:,1,:,:]*0.587+warp_view_list[i][:,2,:,:]*0.114
        mask = torch.clip(cv_gray,0.0,1.0)
        visi_mask.append(mask)
    view_list = [x for i, x in enumerate(view_list) if i not in [4, 13, 22, 31]]

    loss = 0
    for i in range(32):
        loss += criterion(warp_view_list[i],view_list[i])
    return loss



view_IDX =  [36,37,38,39,41,42,43,44,4,13,22,31,49,58,67,76,0,10,20,30,50,60,70,80,8,16,24,32,48,56,64,72]
def noise_warp(disp_list,center):
    tmp = []
    x, y = torch.arange(0, 64), torch.arange(0, 64)
    meshgrid = torch.stack(torch.meshgrid(x, y), -1).unsqueeze(0)
    B,C,H,W = center.shape
    disp = disp_list[0]
    meshgrid = meshgrid.repeat(B, 1, 1, 1).to(disp) # B*H*W*2
    for i in range(32):
        disp = disp_list[i].squeeze(1)

        # assert H==self.patch_size and W==self.patch_size,"size is different!"
        
                                                                                               #############
        u, v = divmod(view_IDX[i], 9) ######## k+2  k+1
        grid = torch.stack([ 
            torch.clip(meshgrid[:,:,:,1]-disp*(4-v),0,W-1),
            torch.clip(meshgrid[:,:,:,0]-disp*(4-u),0,H-1)
        ],-1)/(W-1) *2 -1  # B*H*W*2  归一化到-1，1
        tmp.append(grid_sample(center, grid, align_corners=True))  ######## k+2  k+1
    return tmp


def compute_noise(disp_list,view_list,center):
    noise_mask = []
    view_list = [x for i, x in enumerate(view_list) if i not in [4, 13]]
    warp_view = noise_warp(disp_list,center)
    for i in range(32):
        view1_gray = view_list[i][:,0,:,:]*0.299+view_list[i][:,1,:,:]*0.587+view_list[i][:,2,:,:]*0.114
        view2_gray = warp_view[i][:,0,:,:]*0.299+warp_view[i][:,1,:,:]*0.587+warp_view[i][:,2,:,:]*0.114
        error = torch.abs(view2_gray-view1_gray)
        mask = torch.where(error<0.05,1,0)
        noise_mask.append(1-mask)
    return noise_mask

# def compute_noise(disp_list,view_list):
#     noise_mask = []
#     view_list = [x for i, x in enumerate(view_list) if i not in [4, 13]]
#     for i in range(0,16,2):
#         disp1 = disp_list[i]
#         disp2 = disp_list[i+1]
#         view1 = view_list[i]
#         view2 = view_list[i+1]
#         warp_view1 = noise_warp(disp1,view2)
#         warp_view2 = noise_warp(disp2,view1)
        
#         view1_gray = view1[:,0,:,:]*0.299+view1[:,1,:,:]*0.587+view1[:,2,:,:]*0.114
#         view2_gray = view1[:,0,:,:]*0.299+view2[:,1,:,:]*0.587+view2[:,2,:,:]*0.114
        
#         warp_view1_gray = warp_view1[:,0,:,:]*0.299+warp_view1[:,1,:,:]*0.587+warp_view1[:,2,:,:]*0.114
#         warp_view2_gray = warp_view2[:,0,:,:]*0.299+warp_view2[:,1,:,:]*0.587+warp_view2[:,2,:,:]*0.114
        
#         error1 = torch.abs(warp_view1_gray-view1_gray)
#         error2 = torch.abs(warp_view2_gray-view2_gray)
#         # mask1 = torch.clip(error1,0.0,1.0)
#         # mask1= 1-mask1
#         # mask2 = torch.clip(error2,0.0,1.0)
#         # mask2= 1-mask2
        
#         mask1 = torch.where(error1<0.001,1,0)
#         mask2 = torch.where(error2<0.001,1,0)
#         noise_mask.append(mask1)
#         noise_mask.append(mask2)
#     return noise_mask



# 定义蒸馏损失函数
def get_distillation_loss(student_outputs, teacher_outputs, epoch, alpha):

    # KL散度损失
#     temperature = 8
#     alpha_distillation_loss = 5  # 蒸馏损失的权重，用于平衡蒸馏损失和原始损失之间的关系。
    
#     batch_size = student_outputs.size(0)
#     channel = student_outputs.size(1)
#     student_softmax = nn.functional.softmax(student_outputs.reshape(batch_size,channel,-1) / temperature, dim=2)  #  将输出形状为torch.Size([2, 1, 64, 64])的张量重塑之后再进行softmax
#     teacher_softmax = nn.functional.softmax(teacher_outputs.reshape(batch_size,channel,-1) / temperature, dim=2)
#     kl_div = nn.functional.kl_div(student_softmax.log(), teacher_softmax, reduction='batchmean')  # 计算学生模型的 Softmax 输出和教师模型的 Softmax 输出之间的 KL 散度，并对整个批次的 KL 散度求平均。
    
#     return kl_div * (temperature**2) * alpha_distillation_loss * alpha

    # L1损失
    criterion  = torch.nn.L1Loss()
    L1_loss = criterion(student_outputs,teacher_outputs)
    
    return L1_loss * alpha
