# -*- coding: utf-8 -*-
# @Time    : 2017/12/21 14:52
# @Author  : LeonHardt
# @File    : mat_to_numpy.py

import scipy.io as sio
import numpy as np


def mat_to_numpy(file_name):
    data = sio.loadmat(file_name)['x']
    return data


if __name__ == '__main__':
    """
    Change ginseng mat into numpy
    -----------------------    
    data_mat: sample features
    y_label: sample labels
    """
    # data_mat = mat_to_numpy('ginseng_10feature.mat')
    # np.savetxt('ginseng_x_sample.txt', data_mat, delimiter=',')
    #
    y_label = []
    for sample_number in range(9):
        label = np.ones((35, 1))*sample_number
        if sample_number == 0:
            y_label = label
        else:
            y_label = np.vstack((y_label, label))
    np.savetxt('ginseng_y_label.txt', y_label, delimiter=',')

