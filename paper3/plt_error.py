# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 16:10
# @Author  : LeonHardt
# @File    : plt_error.py

import os
import numpy as np
import matplotlib.pyplot as plt

import glob

methods = ['1NN', 'SVM', 'Tree']
framework = ['BCP', 'CP', 'ICP']
titls = ['(a)', '(b)', '(c)']
linestyles = ('-', '--', '-.', ':')
colors = ('red', 'blue', 'green', 'gray', 'cyan')

fig = plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax = [ax1, ax2, ax3]
for index, method in enumerate(methods):
    ax_temp = ax[index]

    save_path = os.getcwd()+'/summary/'+method+'/*.txt'
    files = glob.iglob(save_path)
    for num, file in enumerate(files):
        data = np.loadtxt(file, delimiter=',')
        ax_temp.plot(data[:, 0], data[:, 1], linestyle='-', color=colors[num],
                     label=framework[num]+'-'+method)
    ax_temp.plot(data[:, 0], data[:, 0], linestyle='--', color='black')
    ax_temp.set_xlabel("Significance", fontsize=10, fontweight='bold')
    ax_temp.set_ylabel("Error rate", fontsize=10, fontweight='bold')
    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, 1)
    # ax_temp.set_xticklabels(fontsize=8, fontweight='bold')
    # ax_temp.set_yticklabels(fontsize=8, fontweight='bold')
    ax_temp.legend(loc='best')
    ax_temp.set_title(titls[index])

plt.show()

