# 学习
# 学生
# 开发时间：2024/4/19 15:50

import numpy as np
import matplotlib.pyplot as plt


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

