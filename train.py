from __future__ import division
from __future__ import print_function

import time
from scipy.io import savemat

import torch

import torch.nn.parallel
import importlib

import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter

from util import create_model, creat_logger
import torchvision.transforms as transforms
from option import TrainOptions
import os


def norm(tensor):  # 对输入的张量进行归一化处理，使得张量中的数值范围被缩放到 [0, 1] 之间。
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))


def main():
    opt = TrainOptions().parse()
    opt.n_epochs_decay = int(3.5 * opt.n_epochs)  # 学习率衰减的周期数 n_epochs_decay = 3.5 * 70 = 245
    opt.lr_decay_iters = opt.n_epochs  # 学习率的衰减步数 lr_decay_iters
    # cudnn related setting
    torch.backends.cudnn.benchmark = True  # 当将此选项设置为True时，CuDNN将根据输入数据的大小和类型选择最适合的卷积算法，以提高性能。这样可以加快训练速度，但可能会导致一些运行时变化，不适用于每个输入大小。仅在输入大小不变的情况下才建议启用此选项。
    torch.backends.cudnn.deterministic = False  # 当将此选项设置为False时，CuDNN将使用非确定性算法，这可能会提高性能，但每次训练的结果可能会有微小差异。如果要确保每次训练得到相同的结果，可以将此选项设置为True，但这可能会降低性能。
    torch.backends.cudnn.enabled = True  # 此选项用于启用CuDNN加速。CuDNN是NVIDIA提供的用于深度神经网络的GPU加速库，通过启用此选项，可以利用CuDNN提供的优化来加速神经网络的训练过程。
    if not opt.debug:  # 这段代码的作用是在非调试模式下进行训练过程中的日志记录和可视化输出的初始化工作。通过创建logger和设置相关目录，可以方便地记录训练过程中的信息并进行可视化分析。
        logger, output_dir, log_dir, tb_dir, time_str = creat_logger(opt,
                                                                     'train')  # 调用creat_logger函数创建一个logger，并返回输出目录、日志目录、TensorBoard目录和时间字符串。
        '''
        在 TensorBoard 中进行可视化：在终端输入tensorboard --logdir=D:\TencentMeeting\vippython\OPAL_main\tb\OPENet\2024-03-04-17-22 或者 OPAL_main/tb/OPENet/2024-03-04-17-22 注意访问的是文件所在的目录而不是文件本身
        '''
        opt.time_str = time_str  # 将时间字符串存储在opt.time_str中。
        writer_dict = {  # 创建一个字典writer_dict，其中包含一个名为writer的SummaryWriter对象，以及用于训练和验证全局步数的变量。
            'writer': SummaryWriter(log_dir=tb_dir),
            # SummaryWriter 是 TensorBoard 的日志写入工具，它用于将数据写入到 TensorBoard 日志文件中，以便后续在 TensorBoard 中进行可视化和分析。在这里，通过指定 log_dir=tb_dir，表示将日志文件保存在 tb_dir 指定的路径下。
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    # 这段代码中定义了一个transforms.Normalize的操作，用于对图像进行标准化处理。具体来说，transforms.Normalize接受两个参数：mean和std，它们分别代表要进行标准化的通道的均值和标准差。
    # 在这个例子中，对图像进行标准化的操作是将每个通道的数值减去均值，然后除以对应通道的标准差。这有助于将图像的每个通道的数值缩放到较小的范围，以便更好地适应神经网络模型的训练。
    normalize = transforms.Normalize(  # std=[0.229, 0.224, 0.225]表示将每个通道的数值除以[0.229, 0.224, 0.225]这个标准差。
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # mean=[0.485, 0.456, 0.406]表示将图像的每个通道的数值减去[0.485, 0.456, 0.406]这个均值。
    )
    # 这行代码通过将字符串'data.' + opt.dataset_file作为参数传递给import_module函数动态地导入了一个名为opt.dataset_file的模块，该模块位于data目录下。
    data_lib = importlib.import_module('data.' + opt.dataset_file)

    train_dataset = data_lib.TrainDataset(opt, True)
    # normalize,])
    valid_dataset = data_lib.ValDataset(opt)
    print('The number of training images = %d' % len(train_dataset))  # 4000
    print('The number of valid images = %d' % len(valid_dataset))  # 4
    # 这段代码使用了PyTorch中的DataLoader类来创建用于训练和验证的数据加载器。具体来说，它创建了一个用于训练数据的train_loader和一个用于验证数据的valid_loader。
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=opt.is_shuffle,  # 是否在每个epoch开始时打乱数据集
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory  # 是否将数据存储在固定内存中，opt.pin_memory是一个布尔值。
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 指定验证数据加载器使用的子进程数目为1。
    )
    model = create_model(opt)
    model.setup()
    total_iters = 0
    best_loss = 1e3
    if not opt.debug:
        writer = writer_dict['writer']
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch    time.time() 函数返回当前时间的时间戳
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        for i, data in enumerate(
                train_loader):  # inner loop within one epoch  这行代码使用了Python的enumerate函数，它可以同时获得迭代对象的索引和值。
            # 在每次迭代中，data 变量会包含从 train_loader 中获取的一个批次（batch）的数据，而 i 则表示该批次数据在整个数据集中的索引值。在该循环中，变量 i从0变化到 len(train_loader)，每次+1
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opt.batch_size  # 整个训练过程中的总迭代次数
            epoch_iter += opt.batch_size  # 当前 epoch 中的迭代次数

            model.set_input(data, epoch)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            # 这段代码是用于在每经过一定步长（opt.print_freq）时打印训练损失并将日志信息保存到磁盘上。
            '''
            如果取余操作的结果为0则打印信息。print_freq为100，total_iters每次循环增加 batch_size，如果 batch_size为2，则每50次循环打印一次信息，如果 batch_size为4，则每25次循环打印一次信息。
            batch_size * len(train_loader) = 4000
            '''
            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()  # 获取当前模型的损失值（losses），一般会包括 L1 损失和平滑度（smoothness）损失。
                t_comp = (
                                     time.time() - iter_start_time) / opt.batch_size  # 计算每个 batch 的完成时间（t_comp），用于评估每个 batch 的处理时间。

                # 这行代码是从 writer_dict 字典中获取当前训练的全局步数 train_global_steps。在训练过程中，全局步数通常用来跟踪整个训练过程中的步数，
                # 包括每个 epoch 中的迭代次数。通过记录全局步数，可以更好地监控训练的进度，并在需要时进行可视化或其他操作。
                global_steps = writer_dict['train_global_steps']
                # 这行代码是将训练过程中的 photometric 损失（一种损失函数）以 scalar （标量）的形式添加到 Tensorboard 的可视化中。
                # 具体来说，writer.add_scalar() 函数用于向 Tensorboard 中添加标量数值，其中 'train_photometric' 是该标量数据的名称，losses['L1'] 是 photometric 损失的数值，
                # global_steps 则是用来在 x 轴上表示该标量数据所对应的全局步数。通过将训练过程中的损失以标量形式添加到 Tensorboard 中，可以方便地监控损失函数的变化趋势，帮助更好地理解模型的训练情况。
                writer.add_scalar('train_photometric', losses['L1'], global_steps)
                smoothloss_msg = 0
                if opt.losses.find('smooth') != -1:  # 判断损失函数列表 opt.losses 中是否包含 'smooth' 这个关键词，如果包含则继续执行下面的逻辑。
                    writer.add_scalar('train_smoothness', losses['smoothness'], global_steps)
                    smoothloss_msg = losses['smoothness']  # 将 smoothloss_msg 更新为 losses['smoothness']，即记录了平滑度损失的数值。
                writer_dict['train_global_steps'] = global_steps + 1
                # 这段代码用于构建一个日志信息字符串 msg，包括了一些训练过程中的关键信息，然后通过 logger 来记录这条日志信息。
                msg = 'Epoch: [{0}/{6}][{1}/{2}]\t' \
                      'speeed: {3}s)\t' \
                      'Loss: {4}\t {5}'.format(epoch, i, len(train_loader), t_comp, losses['L1'], smoothloss_msg,
                                               opt.n_epochs + opt.n_epochs_decay)
                logger.info(msg)
        # 这行代码调用了模型对象的 save_networks 方法，以便将当前网络的参数保存到磁盘上。其中 'latest' 是保存的标识符，可能代表保存最新状态的网络参数。
        model.save_networks('latest')
        # valid
        # syth
        if epoch % 1 == 0:
            model.save_networks(str(epoch))
            mse_log = 0
            gpu_ids = opt.gpu_ids
            device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
            mean_mse = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
            for i,data in enumerate(valid_loader):
            #     if i >= opt.num_val:  # only apply our model to opt.num_test images.
            #         break
                name = data[-1][0]
                model.set_input(data[:-1], epoch)
                model.test()
                visuals = model.get_current_visuals()  # get image results
                output = visuals['output'][0]
                # output = (output - torch.min(output))/(torch.max(output) - torch.min(output))
                # writer.add_image('v35*5_output_'+name, output, epoch)
                label = visuals['label'][0]
                diff = torch.abs(output-label)
                train_bp = torch.where(diff >= 0.07, 1, 0).float()
                mse_x100 = 100 * torch.mean(torch.square(diff))
                bad_pixel = 100 * torch.mean(train_bp)
                writer.add_scalar('v35*5_msex100_'+name, mse_x100, epoch)
                writer.add_scalar('v35*5_bad_pixel_'+name, bad_pixel, epoch)

                diff = (diff - torch.min(diff))/(torch.max(diff) - torch.min(diff))
                output = (output - torch.min(output))/(torch.max(output) - torch.min(output))
                writer.add_image('v35*5_output_'+name, output, epoch)
                writer.add_image('v35*5_diff_'+name, diff, epoch)

                mse_log += mse_x100.item()
                mean_mse += mse_x100
            writer.add_scalar('mean_msex100_' + 'mean_mse', mean_mse / 8, epoch)
            msg = 'mean_mse: {0}'.format(mean_mse / 8)
            logger.info(msg)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        


if __name__ == '__main__':
    main()
