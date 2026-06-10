#####################################################################
# This file is part of the 4D Light Field Benchmark.                #
#                                                                   #
# This work is licensed under the Creative Commons                  #
# Attribution-NonCommercial-ShareAlike 4.0 International License.   #
# To view a copy of this license,                                   #
# visit http://creativecommons.org/licenses/by-nc-sa/4.0/.          #
#####################################################################

from configparser import ConfigParser
import io
import os
import sys

from PIL import Image


import numpy as np


def read_lightfield(data_folder):   # 用于读取光场数据
    params = read_parameters(data_folder)   # 读取光场数据的参数信息
    light_field = np.zeros((params["num_cams_x"], params["num_cams_y"], params["height"], params["width"], 3), dtype=np.uint8)

    views = sorted([f for f in os.listdir(data_folder) if f.startswith("input_") and f.endswith(".png")])
    # 列表推导式通常比使用显式的循环更简洁。从 data_folder 中筛选出以 "input_" 开头且以 ".png" 结尾的文件，并按文件名进行排序，结果保存在 views 列表中。
    for idx, view in enumerate(views):
        fpath = os.path.join(data_folder, view)     # fpath 变量被赋值为由 os.path.join() 函数拼接而成的文件路径，其中 data_folder 是文件夹路径，view 是文件名。
        try:
            img = read_img(fpath)
            light_field[int(idx / params["num_cams_x"]), int(idx % params["num_cams_y"]), :, :, :] = img
        except IOError:
            print("Could not read input file: %s" % fpath) 
            sys.exit()

    return light_field


def read_parameters(data_folder):
    params = dict()     # 创建一个空字典 params，用于存储参数。

    with open(os.path.join(data_folder, "parameters.cfg"), "r") as f:   # 打开了名为 "parameters.cfg" 的配置文件，模式为只读模式
        parser = ConfigParser.ConfigParser()    # 创建了一个配置文件解析器对象。
        parser.readfp(f)    # 读取配置文件的内容
        # 从配置文件中提取相机的内参信息
        section = "intrinsics"
        params["width"] = int(parser.get(section, 'image_resolution_x_px'))
        params["height"] = int(parser.get(section, 'image_resolution_y_px'))
        params["focal_length_mm"] = float(parser.get(section, 'focal_length_mm'))
        params["sensor_size_mm"] = float(parser.get(section, 'sensor_size_mm'))
        params["fstop"] = float(parser.get(section, 'fstop'))
        # 从配置文件中提取相机的外参信息
        section = "extrinsics"
        params["num_cams_x"] = int(parser.get(section, 'num_cams_x'))
        params["num_cams_y"] = int(parser.get(section, 'num_cams_y'))
        params["baseline_mm"] = float(parser.get(section, 'baseline_mm'))
        params["focus_distance_m"] = float(parser.get(section, 'focus_distance_m'))
        params["center_cam_x_m"] = float(parser.get(section, 'center_cam_x_m'))
        params["center_cam_y_m"] = float(parser.get(section, 'center_cam_y_m'))
        params["center_cam_z_m"] = float(parser.get(section, 'center_cam_z_m'))
        params["center_cam_rx_rad"] = float(parser.get(section, 'center_cam_rx_rad'))
        params["center_cam_ry_rad"] = float(parser.get(section, 'center_cam_ry_rad'))
        params["center_cam_rz_rad"] = float(parser.get(section, 'center_cam_rz_rad'))
        # 从配置文件中提取元数据信息
        section = "meta"
        params["disp_min"] = float(parser.get(section, 'disp_min'))
        params["disp_max"] = float(parser.get(section, 'disp_max'))
        params["frustum_disp_min"] = float(parser.get(section, 'frustum_disp_min'))
        params["frustum_disp_max"] = float(parser.get(section, 'frustum_disp_max'))
        params["depth_map_scale"] = float(parser.get(section, 'depth_map_scale'))
        # 除了数值参数外，还有一些字符串参数
        params["scene"] = parser.get(section, 'scene')
        params["category"] = parser.get(section, 'category')
        params["date"] = parser.get(section, 'date')
        params["version"] = parser.get(section, 'version')
        params["authors"] = parser.get(section, 'authors').split(", ")
        params["contact"] = parser.get(section, 'contact')

    return params


def read_depth(data_folder, highres=False):     # data_folder 是数据文件夹的路径，highres 是一个布尔值，用于指示是否读取高分辨率的深度图像，默认为 False。
    fpath = os.path.join(data_folder, "gt_depth_%s.pfm" % ("highres" if highres else "lowres"))
    try:
        data = read_pfm(fpath)
    except IOError:
        print ("Could not read depth file: %s" % fpath)
        sys.exit()
    return data


def read_disparity(data_folder, highres=False):     # 从指定的文件夹中读取视差图像数据
    fpath = os.path.join(data_folder, "gt_disp_%s.pfm" % ("highres" if highres else "lowres"))
    # if highres:
    #     fpath = os.path.join(data_folder, "dino.pfm")
    # else:
    #     fpath = os.path.join(data_folder, "gt_disp_lowres.pfm" )
    try:
        data = read_pfm(fpath)
    except IOError:
        print ("Could not read disparity file: %s" % fpath)
        sys.exit()
    return data


def read_img(fpath):    # 读取指定路径下的图像文件，并将其转换为 RGB 格式的 NumPy 数组
    data = np.array(Image.open(fpath).convert('RGB'))
    return data


def write_hdf5(data, fpath):    # 将输入的数据写入到指定路径的 HDF5 文件中，每个数据项被保存为 HDF5 数据集。
    import h5py
    h = h5py.File(fpath, 'w')   # 创建了一个 HDF5 文件对象 h，使用 'w' 模式表示以写入方式打开文件。
    for key, value in data.iteritems():     # 遍历 data 中的每个键值对
        h.create_dataset(key, data=value)   # 对于每个键值对，创建一个数据集
    h.close()


def write_pfm(data, fpath, scale=1, file_identifier=b"Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)  # 将传入的数据 data 进行上下翻转，然后获取其高度和宽度，并将数据展平为一维数组。
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder    # 获取数据数组 data 的字节序（即数据的存储顺序），并将其赋值给变量 endianess。然后通过 print 函数打印出字节序。
    # print(endianess)
    # 字节序是指在存储多字节数据类型时字节的顺序，有两种常见的字节序：大端序（big-endian）和小端序（little-endian）。大端序是指数据的高位字节存储在低地址处，而小端序则相反，高位字节存储在高地址处。
    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1
    # 如果数据的字节序为小端序（endianess == '<'），或者字节序为与系统字节序相同且系统字节序为小端序（endianess == '=' and sys.byteorder == 'little'），则执行 scale *= -1，即将缩放因子取反。
    with open(fpath, 'wb') as file:     # 打开一个文件对象，使用二进制写入模式（'wb'），并将其赋值给变量 file,依次将文件标识、宽度和高度信息、缩放因子以及数据写入文件中。
        file.write(file_identifier)
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())
        file.write(values)


def read_pfm(fpath, expected_identifier="Pf"):  # PFM 格式是用于存储灰度图像的一种格式
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    with open(fpath, 'rb') as f:
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


def _get_next_line(f):
    next_line = f.readline().rstrip()
    # ignore comments
    while next_line.startswith(b'#'):
        next_line = f.readline().rstrip()
    return next_line