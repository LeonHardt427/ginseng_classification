# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 10:36
# @Author  : LeonHardt
# @File    : plt_error_rate.py

import os
import numpy as np
import matplotlib.pyplot as plt

save_path = os.getcwd()+'/summary/bcp/error-sign.txt'
data = np.loadtxt(save_path, delimiter=',')

# for row in range(data.shape[0]):
#     data[row, 1] = 1-data[row, 1]

print(data)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(data[:, 0], data[:, 1])
plt.show()