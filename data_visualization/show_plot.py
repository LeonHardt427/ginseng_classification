# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 11:10
# @Author  : LeonHardt
# @File    : show_plot.py

import numpy as np
from lda_visualization import lda_visualization_3D, lda_visualization_2D

if __name__ == '__main__':
    X = np.loadtxt('ginseng_x_sample.txt', delimiter=',')
    y = np.loadtxt('ginseng_y_label.txt', delimiter=',', dtype='int8')

    lda_visualization_3D(X, y)

