import importlib

from sklearn import datasets
from model.basemodel import BaseModel

from pathlib import Path
import logging
import time
import numpy as np
import torch

import torch.nn as nn
import numpy



def create_model(opt):
    model_name = opt.model_name  
    model_filename = 'model.model_'+model_name  # 构造了一个模型文件的文件名 model_filename，形式为 'model.model_' + model_name。
    modellib = importlib.import_module(model_filename)  # 使用 Python 的 importlib 模块动态导入这个模型文件，并将其命名为 modellib。
    model = None
    target_model_name = model_name.replace('_', '') + 'model'   # 将模型名称中的下划线去除，再加上 'model'。
    for name, cls in modellib.__dict__.items():     # 遍历导入的模型文件中的每个类，检查是否存在一个类名和 target_model_name 相同（不区分大小写），且是 BaseModel 的子类。
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):  # 使用 .lower() 方法将两个字符串都转换为小写
            model = cls     # 在每次循环迭代中，name 变量存储了当前类的名称，而 cls 变量则存储了当前类的对象。

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance





def creat_logger(opt, mode='train'):
    root_output_dir = Path(opt.output_dir)  # output
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    
    model = opt.model_name  
    final_output_dir = root_output_dir / model  # / 操作符被用来将 root_output_dir 和 model 这两个路径对象连接起来，形成一个新的路径对象，表示最终的输出目录。
    print('=> creating {}'.format(final_output_dir))    # {} 被替换为 final_output_dir 的值
    final_output_dir.mkdir(parents=True, exist_ok=True)  # output/model
    if opt.time_str != '':
        time_str = opt.time_str
    else:
        time_str = time.strftime('%Y-%m-%d-%H-%M')  # 生成一个格式为年-月-日-时-分的时间字符串
        tblog_dir = Path('tb') / model / time_str
        print('=> creating {}'.format(tblog_dir))
        tblog_dir.mkdir(parents=True, exist_ok=True)   # tb_log/model/time

    log_file = '{}_{}.log'.format(mode, time_str)
    final_log_file = final_output_dir / log_file  # output/model/train_time.log

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()    # 创建了一个名为 logger 的日志对象。
    logger.setLevel(logging.INFO)   # 设置日志级别为 INFO，表示只记录 INFO 级别及以上的日志消息。
    console = logging.StreamHandler()   # 创建了一个流处理器对象 console，用于将日志消息输出到控制台。
    logging.getLogger('').addHandler(console)   # 将流处理器添加到根日志对象中，这样日志消息就会同时输出到文件和控制台。

    return logger,  str(final_output_dir), str(final_log_file), str(tblog_dir), time_str


def tensor2im(input_image, mode):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # if image_numpy.shape[0] == 1:  # grayscale to RGB
        #     image_numpy = np.tile(image_numpy, (3, 1, 1))
        # if is_I:
        #     image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
        # else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if mode == 'no':
            ma, mi = image_numpy.max(), image_numpy.min()
            image_numpy = (image_numpy-mi)/(ma-mi)
        image_numpy = image_numpy * 255.0  # post-processing: tranpose and scaling
        # elif label == 'real_BI'  or label == 'fake_BI':
        #     image_numpy = image_numpy * 255.0
        # elif label == 'real_BP' or label == 'fake_BP':
        #     image_numpy = image_numpy * np.pi
        #     ma, mi = image_numpy.max(), image_numpy.min()
        #     image_numpy = (image_numpy - mi)/(ma-mi) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy


def save_current_visual(visu, epoch, iter, writer, phase):
    '''save debug results to tensorboard'''
    modes = ['0,1', 'no', 'no']
    i = 0
    for name, tensor in visu.items():
        # img = tensor2im(name, tensor)
        save_name = phase + '_epoch_' + str(epoch) + '_' + str(iter) +'_'+name

        if len(tensor.shape) == 3:
            dataformats = 'HW'
        else:
            dataformats = 'CHW'
        if modes[i] == 'no': # convert to 0-1
            img_np = tensor[0].detach().cpu().float().numpy()
            ma, mi = img_np.max(), img_np.min()
            img_np = (img_np-mi)/(ma-mi)
            writer.add_image(save_name, img_np, epoch, dataformats=dataformats)
        else:
        # if name == 'center_input':
            writer.add_image(save_name, tensor[0], epoch)  # c*h*w [0,1]   dataformats : CHW, HWC, HW.
        '''https://pytorch.org/docs/1.7.1/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_image'''
        i += 1 
        
        #         writer.add_image(save_name, tensor[0,j].unsqueeze(0), epoch)  # c*h*w [0,1]
            
import os
import cv2
import tifffile as tif
from data.file_io import write_pfm


def save_img(opt, vis, img_name, save_tifdir):  # vis 是一个字典，包含模型的预测结果和真实标签。
    '''save test result to ./results '''
    i = 0
    for vis_name, tensor in vis.items():
        if len(tensor.shape) == 3:      # 如果张量的形状是三维的（例如，高度、宽度、通道），那么它会被扩展为四维张量
            tensor = tensor.unsqueeze(1)
        if i == 1:      # tensor[0,0,:,:] 用于获取张量的第一个样本的第一个通道的二维数组数据。这通常用于获取模型的预测结果或真实标签。
            pred_numpy = tensor[0,0,:,:].cpu().float().numpy()  # 从张量中提取预测结果，并将其转换为 NumPy 数组
            # save_tif = os.path.join(save_tifdir,img_name + '_pred.tiff')    # 构建了要保存的 TIFF 文件的完整路径，其中 save_tifdir 是保存结果的目录路径，img_name 是文件名。
            save_tif = os.path.join(save_tifdir, img_name + '.pfm')
            # 保存为pfm，即将下面的tif.imwrite改为write_pfm即可
            write_pfm(pred_numpy, save_tif)
            # tif.imwrite(save_tif, pred_numpy)   # 用于将 NumPy 数组保存为 TIFF 格式的图像文件。
        elif i==2:  # float() 将数据类型转换为浮点数，以确保数据的类型符合保存图像文件的要求。
            gt_numpy = tensor[0,0,:,:].cpu().float().numpy()    # cpu() 将数据移到 CPU 上进行处理（如果数据之前存储在 GPU 上）。
            save_tif = os.path.join(save_tifdir,img_name + '_gt.tiff')  # 当你需要与其他软件或系统进行交互时，或者需要长期保存图像并确保兼容性时，可以选择保存为.tiff文件。
            tif.imwrite(save_tif, gt_numpy)     # 当你需要保存浮点型图像数据，并且不需要与其他软件进行交互时，可以选择保存为.pfm文件。
        elif i==3:
            pred_numpy = tensor[0,0,:,:].cpu().float().numpy()  # 从张量中提取预测结果，并将其转换为 NumPy 数组
            # save_tif = os.path.join(save_tifdir,img_name + '_pred.tiff')    # 构建了要保存的 TIFF 文件的完整路径，其中 save_tifdir 是保存结果的目录路径，img_name 是文件名。
            save_tif = os.path.join(save_tifdir, img_name + '_coarse.pfm')
            # 保存为pfm，即将下面的tif.imwrite改为write_pfm即可
            write_pfm(pred_numpy, save_tif)
        i += 1

    train_diff = np.abs(pred_numpy - gt_numpy)
    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    train_bp = (train_diff >= 0.07)
    bpr007 = 100 * np.average(train_bp)
    return training_mean_squared_error_x100, bpr007


# hciold is too large
def save_patch(opt, vis, img_name, save_tifdir, info):
    '''save test result to ./results '''
    H = info['H'][0]
    W = info['W'][0]
    out = np.zeros((H, W))
    gt = np.zeros((H, W))
    i = 0
    for vis_name, tensor in vis.items():
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(1)
        if i == 1:
            result = tensor.cpu().float().numpy() 
            out[:512, :512] = result[0,0]
            out[-512:, :512] = result[1,0]
            out[:512, -512:] = result[2,0]
            out[-512:, -512:] = result[3,0]
            save_tif = os.path.join(save_tifdir, img_name + '_pred.tiff')
            tif.imwrite(save_tif, out)
        elif i==2:
            gt_numpy = tensor.cpu().float().numpy() 
            gt[:512, :512] = gt_numpy[0,0]
            gt[-512:, :512] = gt_numpy[1,0]
            gt[:512, -512:] = gt_numpy[2,0]
            gt[-512:, -512:] = gt_numpy[3,0]
            save_tif = os.path.join(save_tifdir, img_name + '_gt.tiff')
            tif.imwrite(save_tif, gt)
        i += 1

    train_diff = np.abs(gt - out)
    training_mean_squared_error_x100 = 100 * np.average(np.square(train_diff))
    train_bp = (train_diff >= 0.07)
    bpr007 = 100 * np.average(train_bp)
    return training_mean_squared_error_x100, bpr007
    