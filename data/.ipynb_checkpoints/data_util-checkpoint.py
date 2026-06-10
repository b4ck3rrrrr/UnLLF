import numpy as np
import imageio
import torch
import tifffile as tif
import os
root = './dataset/hci_dataset/'  # 设置了一个根目录路径变量 root


# 这段代码看起来是用于生成训练数据的函数，主要是针对图像数据进行处理和准备。函数中包含了一些图像处理和数据增强的操作，例如随机选择图像、裁剪、旋转、对比度变换等。
# 另外，代码中也包含了一些条件判断和循环，用于确保生成的训练数据符合特定的要求。
def generate_hci_for_train(traindata_all, traindata_label, input_size, label_size, batch_size,
                           Setting02_AngualrViews, boolmask_img4, boolmask_img6, boolmask_img15, use_v, ouc,
                           num_img):  # ouc：输出通道数  num_img：场景的数量  boolmask_img4, boolmask_img6, boolmask_img15：用于图像反射区域的掩模

    """ initialize image_stack & label 初始化图像堆栈和标签数组 """
    traindata_batch = np.zeros(  # np.zeros() 是 NumPy 库中的一个函数，用于创建指定形状的全零数组。具体来说，np.zeros(shape, dtype) 接受两个参数：
        (input_size, input_size, len(Setting02_AngualrViews), len(Setting02_AngualrViews), ouc),
        dtype=np.float32)  # shape：表示所需数组的形状，可以是一个整数或者一个元组，如 (3, 4) 表示一个形状为 3 行 4 列的二维数组。dtype：表示数组的数据类型，通常为 np.float32、np.int32 等。
    traindata_batch_label = np.zeros((batch_size, label_size, label_size))
    # 通过以上代码，成功创建了用于存储训练数据和标签的数组。
    """ initial variable """
    crop_half1 = int(
        0.5 * (input_size - label_size))  # 0.5*（64-64）=0  crop_half1 的计算结果将为 0,这意味着在裁剪过程中，起始位置将从索引 0 开始，不会有额外的偏移。
    # 通过这段代码，可以得到一个裁剪的边界值，似乎是为了用于裁剪图像时确定裁剪的大小或位置。
    """ Generate image stacks"""
    sum_diff = 0
    valid = 0
    # while循环只选取了某个随机场景的随机裁剪区域64*64，若sum_diff比较小说明中心视图子区域无纹理，一直循环直到选出了有纹理的子区域才停止，然后进行归一化、数据增强（旋转、翻转等）
    while (sum_diff < 0.01 * input_size * input_size or valid < 1):
        """//Variable for gray conversion//"""
        # rand_3color = 0.05 + np.random.rand(3)
        # rand_3color = rand_3color / np.sum(rand_3color)
        # R = rand_3color[0]
        # G = rand_3color[1]
        # B = rand_3color[2]
        R, G, B = 0.299, 0.587, 0.114  # 这些数值实际上是用于灰度图像转换为RGB图像的标准转换系数，称为YCbCr转换系数。

        """
            We use totally 16 LF images,(0 to 15)                   我们总共使用了16个场景的LF图像，（0到15）
            Since some images(4,6,15) have a reflection region,     由于一些图像（4，6，15）具有反射区域，
            We decrease frequency of occurrence for them.           我们降低了它们的发生频率。
            Details in our epinet paper.
        """
        # aa_arr = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23,
        #                     0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23,
        #                     0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23,
        #                     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

        aa_arr = np.arange(num_img)  # 创建了一个从 0 到 num_img-1(20-1) 的整数数组 aa_arr

        image_id = np.random.choice(aa_arr)  # 从范围 [0, num_img-1] 中随机选择一个整数作为图像的编号(场景的序号)
        # 通过这种方法，可以实现在一定范围内均匀地随机选择图像编号，用于后续处理不同图像数据的操作。
        if (len(Setting02_AngualrViews) == 9):
            ix_rd = 0
            iy_rd = 0

        kk = np.random.randint(14)  # 生成一个范围在 [0, 14) 内的随机整数，取值范围是从0到13之间（包括0，不包括14）
        scale = 1
        if (kk < 8):
            scale = 1
        elif (kk < 14):
            scale = 2

        idx_start = np.random.randint(0, 512 - scale * input_size)  # 这些起始位置用于图像处理过程中的裁剪操作。
        idy_start = np.random.randint(0, 512 - scale * input_size)  # 512-64 or 512-128
        valid = 1  # 将 valid 变量设为1，表示当前处理是有效的。
        """
            boolmask: reflection masks for images(4,6,15)
        """
        # 这一段代码的意义何在？
        if (image_id == 4 or 6 or 15):
            if (image_id == 4):
                a_tmp = boolmask_img4
                if (np.sum(a_tmp[  # 这个切片操作是在一个二维数组 a_tmp 上进行的，随机返回一个子区域，大小为64*64。
                           idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                           # 行索引范围：x1:x2:step x1 是开始的行索引（包括 x1）。x2 是结束的行索引（不包括 x2）。列索引范围：y1:y2:step
                           idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0  # step 是行索引的步长。它表示在提取子区域时每隔多少行取一个元素。scale可以取 1 或者 2
                        or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                  idy_start: idy_start + scale * input_size:scale]) > 0):
                    valid = 0  # 总体来说，这段代码根据特定条件判断掩码中两个区域的像素和，如果其中任意一个区域的像素和大于0，则将 valid 设为0；否则 valid 保持为1。
            if (image_id == 6):  # 这三段代码一模一样啊，为什么要重复写三遍啊
                a_tmp = boolmask_img6
                if (np.sum(a_tmp[
                           idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                           idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                        or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                  idy_start: idy_start + scale * input_size:scale]) > 0):
                    valid = 0
            if (image_id == 15):
                a_tmp = boolmask_img15
                if (np.sum(a_tmp[
                           idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                           idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                        or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                  idy_start: idy_start + scale * input_size:scale]) > 0):
                    valid = 0

        if (valid > 0):
            # 代码中的计算涉及对选定的像素数据进行颜色通道加权处理，最终生成中心视图的灰度图（1*64*64*1*1*1 → 64*64），并进行归一化处理。
            image_center = (1 / 255) * np.squeeze(  # np.squeeze() 是 NumPy 库中的一个函数，用于移除数组中的单维条目，即将维数为 1 的维度去除。
                R * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 0].astype('float32') +
                G * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 1].astype('float32') +
                B * traindata_all[image_id, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, 4 + ix_rd, 4 + iy_rd, 2].astype('float32'))
            # 将刚刚计算得到的image_center的每个像素与中心像素做差取绝对值并求和，计算图像中心点与特定位置像素值之间的差异总和，评估图像中心点周围像素值的变化情况。
            sum_diff = np.sum(
                np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))
            # 将全部的图像进行灰度图变换
            if ouc == 1:  # np.expand_dims(..., -1)：在最后一个维度上添加一个维度，将其扩展为 (64*64*9*9 → 64*64*9*9*1) 的形状。
                traindata_batch = np.expand_dims(np.squeeze(  # np.squeeze(...)：压缩维度，去除维度为1的维度。(1*64*64*9*9*1 → 64*64*9*9)
                    R * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 0].astype(
                        'float32') +    # 为什么要这样啊？（image_id:image_id + 1）直接和上面一样只用 image_id 就好了呀
                    G * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 1].astype(
                        'float32') +
                    B * traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                        idy_start: idy_start + scale * input_size:scale, :, :, 2].astype(
                        'float32')), -1)
            else:
                # 裁剪64*64的窗口作为训练数据（64*64*9*9*3）并没有batch维度
                traindata_batch = np.squeeze(
                    traindata_all[image_id:image_id + 1, idx_start: idx_start + scale * input_size:scale,
                    idy_start: idy_start + scale * input_size:scale, :, :, :].astype(
                        'float32'))
            '''
                traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
            '''
            traindata_batch_label[0, :, :] = (1.0 / scale) * traindata_label[image_id,
                                                             idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                             idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]
            # traindata_batch_label[0, :, :]：选择 traindata_batch_label 中第一个样本的所有行和列（即整个二维数组）。仅仅存在一个样本，因为前面的batch定义为了1
            # (1.0 / scale)：计算缩放因子的倒数，用于对选定的标签数据进行反向缩放操作。
            # 从 traindata_label 中选择特定区域的标签数据，并根据给定的步长 scale 进行采样，以实现缩放。
    traindata_batch = np.float32((1 / 255) * traindata_batch)
    # 由于图像数据通常存储在 0 到 255 的范围内，除以 255 可以将像素值缩放到 0 到 1 之间，实现归一化操作。
    '''
    data argument
    数据增强是一种数据扩充技术，指的是利用有限的数据创造尽可能多的利用价值。因为虽然现在各种任务的公开数据集有很多，但是其实数据量也远远不够，
    而公司或者学术界去采集、制作这些数据的成本其实是很高的，像人工标注数据的任务量就很大，因此，只能通过一些方法去更好的利用现有的成本。
    传统数据增强方式有随机翻转、旋转、裁剪、变形缩放、添加噪声、颜色扰动等等。
    数据增强的作用：
    1.避免过拟合。当数据集具有某种明显的特征，例如数据集中图片基本在同一个场景中拍摄，使用Cutout方法和风格迁移变化等相关方法可避免模型学到跟目标无关的信息。
    2.提升模型鲁棒性，降低模型对图像的敏感度。当训练数据都属于比较理想的状态，碰到一些特殊情况，如遮挡，亮度，模糊等情况容易识别错误，对训练数据加上噪声，掩码等方法可提升模型鲁棒性。
    3.增加训练数据，提高模型泛化能力。
    4.避免样本不均衡。在工业缺陷检测方面，医疗疾病识别方面，容易出现正负样本极度不平衡的情况，通过对少样本进行一些数据增强方法，降低样本不均衡比例。
     比较常用的几何变换方法主要有：翻转，旋转，裁剪，缩放，平移，抖动。值得注意的是，在某些具体的任务中，当使用这些方法时需要主要标签数据的变化，如目标检测中若使用翻转，则需要将gt框进行相应的调整。
     比较常用的像素变换方法有：加椒盐噪声，高斯噪声，进行高斯模糊，调整HSV对比度，调节亮度，饱和度，直方图均衡化，调整白平衡等。
    '''
    # contrast
    gray_rand = 0.4 * np.random.rand() + 0.8  # 随机生成一个介于 0.8 到 1.2 之间的浮点数作为对比度的倍数。    np.random.rand(): 这个函数会生成一个 0 到 1 之间的随机浮点数。
    traindata_batch = pow(traindata_batch, gray_rand)  # 将 traindata_batch 中的每个像素值提升到 gray_rand 次幂，从而改变图像的对比度。
    rot = True  # 设置是否进行图像旋转的标志位。
    # 旋转这一部分的代码真是不理解
    if not rot:  # not rot 的意思是 rot 的取值为假（False）时执行以下代码块。
        traindata_batch = np.transpose(traindata_batch, (
        4, 0, 1, 2, 3))  # 调用 NumPy 库中的 np.transpose 函数对 traindata_batch 进行维度转换操作，按照 (4, 0, 1, 2, 3) 的顺序重新排列。
    else:
        rotation_rand = np.random.randint(0, 5)
        # # h*w*9*9*3   1*h*w
        if rotation_rand == 0:  # 这段代码根据随机选择的旋转方式，在不同维度上对训练数据进行旋转操作，同时保持标签数据的一致性，以便在训练过程中使用旋转后的数据增强模型的泛化能力。
            traindata_batch = np.transpose(traindata_batch, (4, 0, 1, 2, 3))
        elif rotation_rand == 1:  # 90    np.copy 是 NumPy 库中用于创建数组的副本的函数。当使用 np.copy(array) 时，它会返回一个与 array 具有相同数据的新数组，但是这两个数组在内存中是独立的。换句话说，对新数组的更改不会影响原始数组，反之亦然。
            traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch))  # 这行代码将 traindata_batch 沿着第一个维度（通常是高度）进行 90 度旋转
            traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 1, (
            2, 3)))  # w*h*9*9*3  1: 表示顺时针旋转 90 度。(2, 3): 指定对数组的第二和第三个维度（通常是宽度和颜色通道）进行旋转操作。
            traindata_batch = np.transpose(traindata_batch_tmp5, (4, 0, 1, 2, 3))
            traindata_label_tmp6 = np.copy(
                np.rot90(traindata_batch_label[0, :, :]))  # 从 traindata_batch_label 数组中选择第一个元素进行旋转操作
            traindata_batch_label[0, :, :] = traindata_label_tmp6
        elif rotation_rand == 2:  # 180
            traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch, 2))
            traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 2, (2, 3)))  # 数字 2 表示旋转的次数
            traindata_batch = np.transpose(traindata_batch_tmp5, (4, 0, 1, 2, 3))
            traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :], 2))
            traindata_batch_label[0, :, :] = traindata_label_tmp6
        elif rotation_rand == 3:  # 270
            traindata_batch_tmp6 = np.copy(np.rot90(traindata_batch, 3))
            traindata_batch_tmp5 = np.copy(np.rot90(traindata_batch_tmp6, 3, (2, 3)))  # 3表示进行三次逆时针旋转
            traindata_batch = np.transpose(traindata_batch_tmp5, (4, 0, 1, 2, 3))
            traindata_label_tmp6 = np.copy(np.rot90(traindata_batch_label[0, :, :], 3))
            traindata_batch_label[0, :, :] = traindata_label_tmp6
        else:  # flip 这一段代码是个啥呀，完全看不懂
            traindata_batch_tmp = np.copy(np.rot90(np.transpose(traindata_batch, (1, 0, 4, 2, 3))))
            traindata_batch = np.copy(np.transpose(traindata_batch_tmp[:, :, :, ::-1], (2, 0, 1, 3, 4)))  # 3*w*h*9*9

            traindata_batch_label_tmp = np.copy(np.rot90(np.transpose(traindata_batch_label[0, :, :], (1, 0))))
            traindata_batch_label[0, :, :] = traindata_batch_label_tmp  # 1*w*h

    traindata_all = torch.FloatTensor(traindata_batch)
    label_all = torch.FloatTensor(traindata_batch_label)
    return traindata_all, label_all

'''
下面这段加载数据的代码只是用来加载数据的，直接将数据集中的所有数据全部加载即可
'''
root = './dataset/'
# 问：无监督网络为什么要加载深度图（真实标签）？
# 答：为了统一训练、验证和测试这三者的数据加载方法（因为验证的时候需要根据真实值来衡量性能，而训练和测试的时候加载不到真实值也无所谓，因为加载到之后也不会进行使用）。

def load_hci(dir_LFimages):     # 用于加载 HCI 数据集中的图像和标签数据。
    tmp = np.float32(imageio.imread(root+dir_LFimages[0]+'/input_Cam000.png'))
    h,w,_ = tmp.shape
    del tmp
    traindata_all=np.zeros((len(dir_LFimages), h, w, 9, 9, 3),np.uint8)     # 此处 len(dir_LFimages) = 20
    traindata_label=np.zeros((len(dir_LFimages), h, w),np.float32)
    image_id = 0    # 表示场景的序列号，一共20个场景，每个场景都有一个视差图作为标签
    for dir_LFimage in dir_LFimages:    # 开始遍历指定的 dir_LFimages 目录列表中的每个场景。
        print(dir_LFimage)  # 先在控制台输出场景路径
        for i in range(81):     # 对于每个目录，使用嵌套的循环加载该目录下的 81 张图像文件，并将它们存储在 traindata_all 数组中的适当位置。
            try:
                tmp = np.float32(imageio.imread(root+dir_LFimage+'/input_Cam0%.2d.png' % i))  # load LF images(9x9)
            except:     # 如果无法加载某个图像文件，则打印错误信息。
                print(root+dir_LFimage+'/input_Cam0%.2d.png..does not exist')
                tmp = np.zeros((h, w, 3))   # 自己根据下面的视差图代码写了一行代码来避免报错
            traindata_all[image_id, :, :, i // 9, i - 9 * (i // 9), :] = tmp
            del tmp
        try:
            tmp = np.float32(read_pfm(root+dir_LFimage+'/gt_disp_lowres.pfm'))  # load LF disparity map
        except:     # 继续加载该目录下的标签文件，如果无法加载标签文件则创建一个全零数组。
            print(root+dir_LFimage+'/gt_disp_lowres.pfm..does not exist')
            tmp = np.zeros((h, w))
        traindata_label[image_id,:,:] = tmp     # 将标签数据存储在 traindata_label 数组中的适当位置。
        del tmp
        image_id = image_id+1
    return traindata_all, traindata_label


# def load_hci(dir_LFimages):  # 用于加载 HCI 数据集中的图像和标签数据。
#     traindata_all = np.zeros((len(dir_LFimages), 512 ,512, 9, 9, 3), np.uint8)  # 此处 len(dir_LFimages) = 20
#     traindata_label = np.zeros((len(dir_LFimages), 512, 512), np.float32)
#     image_id = 0  # 表示场景的序列号，一共20个场景，每个场景都有一个视差图作为标签
#     for dir_LFimage in dir_LFimages:  # 开始遍历指定的 dir_LFimages 目录列表中的每个场景。
#           # 先在控制台输出场景路径
#         for i in range(81):  # 对于每个目录，使用嵌套的循环加载该目录下的 81 张图像文件，并将它们存储在 traindata_all 数组中的适当位置。
#             try:
#                 tmp = np.float32(imageio.imread(real_world_root + dir_LFimage + '/input_Cam%.3d.png' % i))  # load LF images(9x9) %.2d 中的 .2 表示最少输出两位数字，不足两位时前面补零。

#                 # tmp = np.float32(imageio.imread(root + dir_LFimage + '/input_Cam0%.2d.png' % i))
#             except:  # 如果无法加载某个图像文件，则打印错误信息。
                
#                 tmp = np.zeros((512, 512, 3))  # 自己根据下面的视差图代码写了一行代码来避免报错

#             traindata_all[image_id, :, :, i // 9, i - 9 * (i // 9), :] = tmp
#             del tmp
#         try:
#             tmp = np.float32(read_pfm(real_world_root + dir_LFimage + '/gt_disp_lowres.pfm'))  # load LF disparity map
#         except:  # 继续加载该目录下的标签文件，如果无法加载标签文件则创建一个全零数组。
            
#             tmp = np.zeros((512, 512))
#         traindata_label[image_id, :, :] = tmp  # 将标签数据存储在 traindata_label 数组中的适当位置。
#         del tmp
#         image_id = image_id + 1
#     return traindata_all, traindata_label
    # 返回一个6维的列表traindata_all，包含了所有用于训练的数据，以及返回了一个3维列表traindata_label，包含了所有场景的视差图


def read_pfm(fpath, expected_identifier="Pf"):  # 用于读取 PFM（Portable FloatMap）格式文件数据的函数
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html
    # The identifier line contains the characters "PF" or "Pf". PF means it's a color PFM. Pf means it's a grayscale PFM.
    # 在 PFM 文件格式中，通常文件的开头会包含一个标识符，用来表明文件的类型或格式。
    def _get_next_line(f):  # 用于从文件中读取下一行数据，并跳过以 # 开头的注释行。
        next_line = f.readline().decode(
            'utf-8').rstrip()  # 使用f.readline()方法从文件中读取下一行内容。使用decode('utf-8')将读取的内容按照UTF-8编码进行解码，确保读取到的内容是字符串。
        # ignore comments                                     使用rstrip()方法去除行末的换行符和空白字符。
        while next_line.startswith('#'):  # 检查读取的行是否以#开头，如果是注释行，则继续读取下一行内容，直到读取到不是以#开头的行为止。
            next_line = f.readline().rstrip()
        return next_line  # 最后返回整理过的下一行内容。

    # 这段代码使用了 Python 中的 with 语句，打开了一个指定路径的文件，并将它赋值给变量 f。由于我们打开的是一个二进制文件，因此指定的读取模式为 'rb'，而不是普通文本文件的读取模式 'r'。
    with open(fpath, 'rb') as f:  # 打开指定路径的 PFM 文件，并逐行读取文件内容。
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))
        return data
