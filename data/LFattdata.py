from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torchvision import transforms

import numpy as np
import torch
from torch.utils.data import Dataset
import math
import os
import imageio  # 因为在data_util中已经导入了imageio、torch、numpy，而且本文件又导入了全部的data_util，所以此处显示灰色
from .data_util import *

mytransforms = transforms.Compose([transforms.ToTensor()])

root = './dataset/hci_dataset/'
# root = './dataset/hci_blender/image/'

class TrainDataset(Dataset):    # 这段代码定义了一个名为 TrainDataset 的类，该类继承自 torch.utils.data.Dataset。主要用于创建用于训练的数据集对象。
    def __init__(self, opt, istrain=True):  # __init__ 方法用于初始化数据集，设置一些参数，并加载数据。
        self.opt = opt  # 该变量保存了一些选项（options），可能是用于配置数据集加载的参数。
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])  # 该变量是一个包含 9 个元素的 NumPy 数组，表示视角的索引。它指定了用于构建训练数据的角度视图。
        self.input_size = 64    # 输入图像的尺寸，这里设置为 64x64 像素。
        self.label_size = self.input_size   # 标签图像的尺寸，与输入图像的尺寸相同。
        self.use_v = 9  # 表示使用的视角数量，这里设置为 9。
        self.inc = 3    # 表示每个视角之间的增量，这里设置为 3。
        print('Load hci data...')   # 数据加载过程中，会加载一系列目录中的图像数据，其中包括训练数据和标签数据。
        dir_LFimages = [    # dir_LFimages 是一个包含多个字符串元素的列表，每个字符串表示一个目录的路径。
            'additional/antinous', 'additional/boardgames', 'additional/dishes', 'additional/greek',
            'additional/kitchen', 'additional/medieval2', 'additional/museum', 'additional/pens',
            'additional/pillows', 'additional/platonic', 'additional/rosemary', 'additional/table',
            'additional/tomb', 'additional/tower', 'additional/town', 'additional/vinyl',]

        with_valid = False
        if with_valid:
            dir_LFimages += ['stratified/dots', 'stratified/pyramids', 'stratified/stripes', 'stratified/backgammon']
            # 'test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami']
        self.traindata_all, self.traindata_label = load_hci(dir_LFimages)

        print('Load training data... Complete')

        # 对遮挡有了新的理解：对于像玻璃这样的透明物体，它的深度会错误地记录为透过玻璃，在玻璃之后物体的深度，或者记录了玻璃上反光物体的深度，对于反光的不锈钢，进行像素匹配的时候可能匹配不到或者记录的是反光物体的深度。
        # load invalid regions from training data (ex. reflective region)   在加载数据时，还会加载一些无效区域的遮罩数据，用于标记图像中的无效区域。
        boolmask_img4 = imageio.imread(root + 'additional_invalid_area/kitchen/input_Cam040_invalid_ver2.png', pilmode='RGBA')   # 由于这里的无效数据集找不到，因此自己创建了文件夹，然后将场景4、6、15的中心视图放了进去
        boolmask_img6 = imageio.imread(root + 'additional_invalid_area/museum/input_Cam040_invalid_ver2.png')    # 无效区域指的就是反射区域，场景4中存在不锈钢和玻璃罩反射，场景6中存在玻璃反射，场景15中的
        boolmask_img15 = imageio.imread(root + 'additional_invalid_area/vinyl/input_Cam040_invalid_ver2.png')    # 不锈钢音乐播放器反射了来自书的光，接下来要生成 mask 遮挡掉无效区域
        # 加载的图像有3个维度，分别为 H、W、C（RGB 3 通道），加了一句代码 pilmode="RGBA 之后，有了透明度通道，但是值全为255    PNG是一种使用RGBA的图像格式,32位PNG在24位基础上增加了8位透明通道，因此可展现256级透明程度。
        # print('boolmask_img4的shape:', boolmask_img4.shape)
        # print('boolmask_img4的dtype:', boolmask_img4.dtype)
        # print(boolmask_img4)
        # print(np.transpose(boolmask_img6,(2,0,1)))
        self.boolmask_img4 = 1.0*boolmask_img4[:,:,3] > 0    # [:,:,3] 表示对图像数据的所有行和列进行选择，并且只选择透明度通道的数值。这样得到的结果是一个二维数组，其中每个元素代表相应位置像素的透明度值。
        self.boolmask_img6 = 1.0*boolmask_img6[:,:,3] > 0    # 将 boolmask_img4 的 alpha 通道（第四个通道）的像素值大于 0 的部分设为真，其余部分设为假。这样可以得到一个布尔掩码，0表示图像中的无效区域。
        self.boolmask_img15 = 1.0*boolmask_img15[:,:,3] > 0   # 这里索引“3”超出了范围，为了能够运行，我改成了“2”。完全透明的部分为无效区域，用mask遮挡掉。
        # print(self.boolmask_img4)
        # 在处理图像数据时，通常使用 RGBA 表示每个像素的颜色信息。RGBA 表示红色（Red）、绿色（Green）、蓝色（Blue）和透明度（Alpha）四个通道。这些通道分别表示像素的颜色信息和透明度信息。
        # 透明度通道（Alpha）：表示像素的透明度，通常用于指示像素的不透明度程度，值越大表示越不透明。0 表示完全透明的像素，即该像素不可见。1 表示完全不透明的像素，即该像素完全可见。
        # 在PS里，建立上下两个图层，我们设定上面图层的透明度为40%，那么混合后的图像每个像素的R、G、B分量值就可以用下面这个经典的线性插值方程计算出来了：
        # α×A + (1-α) × B   方程中的A是上层图像每个像素里RGB的分量值，B是下层图像对应位置像素里RGB的分量值，α是透明度，用百分比表示，计算结果就是混合后图像对应位置像素RGB的分量值了
        # 这个过程就像给上层图层加了一个透明度调节器一样，这个透明度调节器只能让上层图层的某个区域显示40%
        # 如果一种颜色的 alpha 值为 0，那么它就是不可见的，RGB 值是多少并不重要。毕竟看不见的红色和看不见的黑色是一样的。
        # 当使用 imageio 读取 PNG 图像时，如果图像包含 Alpha 通道（即 RGBA 图像），则透明度信息会被正确地读取并保存在图像数据中。

        self.num_img = len(dir_LFimages)    # 场景的数量：20

    def __len__(self):  # __len__ 方法返回数据集的长度，这里固定返回值为 4000。
        return 500

    def __getitem__(self, index):   # __getitem__ 方法根据给定的索引 index，调用函数 generate_hci_for_train 来生成训练数据。
        # if index > 2000:
        return generate_hci_for_train(self.traindata_all, self.traindata_label,
                                                                self.input_size,self.label_size,1,  # batch_size=1      ouc = inc
                                                                self.Setting02_AngualrViews,
                                                                self.boolmask_img4,self.boolmask_img6,self.boolmask_img15, self.use_v, self.inc, self.num_img)


# hci
class ValDataset(Dataset):  # 这段代码定义了一个名为 ValDataset 的数据集类，用于加载验证数据集。
    def __init__(self, opt):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = opt.input_size

        print('Load test data...')
        # ## hci
    

        
        self.dir_LFimages_hci=['training/boxes', 'training/cotton', 'training/dino', 'training/sideboard','stratified/stripes','stratified/pyramids','stratified/dots','stratified/backgammon']
        self.valdata_hci, self.label_hci = load_hci(self.dir_LFimages_hci)
        # 采用YUV色彩空间的重要性是它的亮度信号Y和色度信号U、V是分离的。如果只有Y信号分量而没有U、V分量，那么这样表示的图像就是黑白灰度图像。
        # 彩色电视采用YUV空间正是为了用亮度信号Y解决彩色电视机与黑白电视机的兼容问题，使黑白电视机也能接收彩色电视信号。
        # Y= 0.299*R + 0.587*G + 0.114*B    YCbCr模型来源于YUV模型，YCbCr是 YUV 颜色空间的偏移版本。
        if opt.input_c == 1:    # 将彩色图转换为灰度图（单通道图像）并作为一个新的维度
            self.valdata_hci = np.expand_dims(self.valdata_hci[:,:,:,:,:,0] * 0.299 + self.valdata_hci[:,:,:,:,:,1] * 0.587 + self.valdata_hci[:,:,:,:,:,2] * 0.114, -1)
        print('Load test data... Complete')     # 使用 np.expand_dims(..., -1) 添加一个新的维度，将计算得到的灰度图数据拓展为最后一个维度

    def __len__(self):
        return len(self.dir_LFimages_hci)

    def __getitem__(self, index):
        center = self.opt.use_views // 2    # // 执行整数除法并返回整数部分，而 / 执行普通除法并返回浮点数结果。
        valid_data = self.valdata_hci / 255.    # 对验证数据进行处理，将其归一化至 0 到 1 范围。
        name = self.dir_LFimages_hci[index].split('/')  # 将路径字符串按照 / 进行分割，比如'training/boxes'分割为'training','boxes'，然后存储到name列表中
        # np.transpose() 是 NumPy 库中的函数，用于对数组进行转置操作。它可以改变数组的维度顺序，使得原数组的某些轴在新数组的不同位置上。
        # 对于（H*W*C）的图像来说，直接在控制台展示的话，每一行代表的是三个通道的像素值，进行（2，0，1）转置后，在控制台显示的是，每一行就是行像素，每一列就是列像素，然后三个通道分开展示，进行后续的图像处理非常方便。
        valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))  # 此处的center应该是用来区分9*9的子孔径图像和5*5的子孔径图像
        test_label = torch.FloatTensor(self.label_hci[index, :,:])  # torch.FloatTensor() 是 PyTorch 中用于将数据转换为浮点型张量（Tensor）的函数。
        return valid_data, test_label, name[1]  # 81*512*512  512*512  array


'''
直接将验证的代码 ValDataset 复制过来作为测试的代码     在测试的时候不需要裁剪，直接将整张图片作为输入即可，操作流程为：加载所有数据→选取子孔径范围→加载权重→准备输入→经过模型获得结果
测试的时候记得将最新的权重移出来，训练的时候放回去
'''

class TestDataset(Dataset):  # 这段代码定义了一个名为 ValDataset 的数据集类，用于加载验证数据集。
    def __init__(self, opt):
        self.opt = opt
        self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
        self.input_size = opt.input_size

        print('Load test data...')
        # ## hci
        # self.dir_LFimages_hci=['training/dino']
        self.dir_LFimages_hci = [f'buildings/buildings_{i}_eslf' for i in range(1, 20)]
        # self.dir_LFimages_hci=['reflective/reflective_1_eslf','reflective/reflective_2_eslf','reflective/reflective_3_eslf','reflective/reflective_4_eslf','reflective/reflective_5_eslf','reflective/reflective_6_eslf','reflective/reflective_7_eslf','reflective/reflective_8_eslf','reflective/reflective_9_eslf','reflective/reflective_10_eslf','reflective/reflective_11_eslf','reflective/reflective_12_eslf','reflective/reflective_13_eslf','reflective/reflective_14_eslf','reflective/reflective_15_eslf','reflective/reflective_16_eslf','reflective/reflective_17_eslf','reflective/reflective_18_eslf','reflective/reflective_19_eslf']
        # self.dir_LFimages_hci=['1024/Lego Bulldozer']
        self.valdata_hci, self.label_hci = load_hci(self.dir_LFimages_hci)
        # 采用YUV色彩空间的重要性是它的亮度信号Y和色度信号U、V是分离的。如果只有Y信号分量而没有U、V分量，那么这样表示的图像就是黑白灰度图像。
        # 彩色电视采用YUV空间正是为了用亮度信号Y解决彩色电视机与黑白电视机的兼容问题，使黑白电视机也能接收彩色电视信号。
        # Y= 0.299*R + 0.587*G + 0.114*B    YCbCr模型来源于YUV模型，YCbCr是 YUV 颜色空间的偏移版本。
        if opt.input_c == 1:    # 将彩色图转换为灰度图（单通道图像）并作为一个新的维度
            self.valdata_hci = np.expand_dims(self.valdata_hci[:,:,:,:,:,0] * 0.299 + self.valdata_hci[:,:,:,:,:,1] * 0.587 + self.valdata_hci[:,:,:,:,:,2] * 0.114, -1)
        print('Load test data... Complete')     # 使用 np.expand_dims(..., -1) 添加一个新的维度，将计算得到的灰度图数据拓展为最后一个维度

    def __len__(self):
        return len(self.dir_LFimages_hci)

    def __getitem__(self, index):
        center = self.opt.use_views // 2    # // 执行整数除法并返回整数部分，而 / 执行普通除法并返回浮点数结果。
        valid_data = self.valdata_hci / 255.    # 对验证数据进行处理，将其归一化至 0 到 1 范围。
        name = self.dir_LFimages_hci[index].split('/')  # 将路径字符串按照 / 进行分割，比如'training/boxes'分割为'training','boxes'，然后存储到name列表中
        # np.transpose() 是 NumPy 库中的函数，用于对数组进行转置操作。它可以改变数组的维度顺序，使得原数组的某些轴在新数组的不同位置上。
        # 对于（H*W*C）的图像来说，直接在控制台展示的话，每一行代表的是三个通道的像素值，进行（2，0，1）转置后，在控制台显示的是，每一行就是行像素，每一列就是列像素，然后三个通道分开展示，进行后续的图像处理非常方便。
        valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))  # 此处的center应该是用来区分9*9的子孔径图像和5*5的子孔径图像
        test_label = torch.FloatTensor(self.label_hci[index, :,:])  # torch.FloatTensor() 是 PyTorch 中用于将数据转换为浮点型张量（Tensor）的函数。
        return valid_data, test_label, name[1]  # 81*512*512  512*512  array
# real_world_root = './dataset/hci_dataset/432/'
# class TestDataset(Dataset):
#     def __init__(self, opt):
#         self.opt = opt
#         self.Setting02_AngualrViews = np.array([0,1,2,3,4,5,6,7,8])
#         self.input_size = opt.input_size

#         print('Load test data...')
#         # ## hci
#         # self.dir_LFimages_hci=['additional/dishes']
#         # self.dir_LFimages_hci=['432/Black_Fence']
#         self.dir_LFimages_hci = []
#         for filename in os.listdir(real_world_root):
#             self.dir_LFimages_hci.append(filename)
#         print(self.dir_LFimages_hci)
#         self.valdata_hci, self.label_hci = load_hci(self.dir_LFimages_hci)
        
#         # 采用YUV色彩空间的重要性是它的亮度信号Y和色度信号U、V是分离的。如果只有Y信号分量而没有U、V分量，那么这样表示的图像就是黑白灰度图像。
#         # 彩色电视采用YUV空间正是为了用亮度信号Y解决彩色电视机与黑白电视机的兼容问题，使黑白电视机也能接收彩色电视信号。
#         # Y= 0.299*R + 0.587*G + 0.114*B    YCbCr模型来源于YUV模型，YCbCr是 YUV 颜色空间的偏移版本。
#         if opt.input_c == 1:    # 将彩色图转换为灰度图（单通道图像）并作为一个新的维度
#             self.valdata_hci = np.expand_dims(self.valdata_hci[:,:,:,:,:,0] * 0.299 + self.valdata_hci[:,:,:,:,:,1] * 0.587 + self.valdata_hci[:,:,:,:,:,2] * 0.114, -1)
#         print('Load test data... Complete')     # 使用 np.expand_dims(..., -1) 添加一个新的维度，将计算得到的灰度图数据拓展为最后一个维度

#     def __len__(self):
#         return len(self.dir_LFimages_hci)

#     def __getitem__(self, index):
#         # center = self.opt.use_views // 2    # // 执行整数除法并返回整数部分，而 / 执行普通除法并返回浮点数结果。
#         an_crop = math.ceil((14 - self.opt.use_views) / 2)    # 17为加载视图的数量
#         '''
#         用上面这行代码取代center那行代码
#         '''
#         valid_data = self.valdata_hci / 255.    # 对验证数据进行处理，将其归一化至 0 到 1 范围。
#         name = self.dir_LFimages_hci[index].split('/')  # 将路径字符串按照 / 进行分割，比如'training/boxes'分割为'training','boxes'，然后存储到name列表中
#         # np.transpose() 是 NumPy 库中的函数，用于对数组进行转置操作。它可以改变数组的维度顺序，使得原数组的某些轴在新数组的不同位置上。
#         # 对于（H*W*C）的图像来说，直接在控制台展示的话，每一行代表的是三个通道的像素值，进行（2，0，1）转置后，在控制台显示的是，每一行就是行像素，每一列就是列像素，然后三个通道分开展示，进行后续的图像处理非常方便。
#         # valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, 4-center:5+center, 4-center:5+center,:], (4,0,1,2,3)))  # 此处的center应该是用来区分9*9的子孔径图像和5*5的子孔径图像
#         valid_data = torch.FloatTensor(np.transpose(valid_data[index, :, :, an_crop:an_crop + self.opt.use_views, an_crop:an_crop + self.opt.use_views, :], (4, 0, 1, 2, 3)))
#         '''
#         上面这行代码可以对加载的数据进行中心子孔径视图选择，以center为中心选取附近的视图
#         '''
#         test_label = torch.FloatTensor(self.label_hci[index, :,:])  # torch.FloatTensor() 是 PyTorch 中用于将数据转换为浮点型张量（Tensor）的函数。
#         return valid_data, test_label, name[0]  # 81*512*512  512*512  array
