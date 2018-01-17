# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 16:10
# @Author  : LeonHardt
# @File    : plt_error.py

import os
import numpy as np
import matplotlib.pyplot as plt

underlying = '1NN'

methods = ['CP', 'ICP', 'BCP']
linestyles = ('-', '--', '-.', ':')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

fig = plt.figure()
ax = plt.subplot(1, 1, 1)

for index, method in enumerate(methods):
    save_path = os.getcwd()+'/summary/'+method+'/significance_error_'+underlying+'_'+method+'.txt'
    data = np.loadtxt(save_path, delimiter=',')
    ax.plot(data[:, 0], data[:, 1], linestyle=linestyles[index], color=colors[index], label=method)
ax.legend(methods)

plt.show()

