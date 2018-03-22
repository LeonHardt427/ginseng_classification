# -*- coding: utf-8 -*-
# @Time    : 2018/3/22 14:58
# @Author  : LeonHardt
# @File    : efficiency_plot.py

import os
import numpy as np
import matplotlib.pyplot as plt

import glob

methods = ['1NN', 'SVM', 'Tree']
framework = ['BCP', 'CP', 'ICP']
titls = ['(a)', '(b)', '(c)']
linestyles = ('-', '--', '-.', ':')
colors = ('red', 'blue', 'green', 'gray', 'cyan')

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

ax = [ax1, ax2, ax3]
for index, method in enumerate(methods):
    ax_temp = ax[index]

    save_path = os.getcwd()+'/efficiency/'+method+'/*.txt'
    files = glob.iglob(save_path)
    for num, file in enumerate(files):
        data = np.loadtxt(file, delimiter=',')
        ax_temp.plot(data[1:, 0], data[1:, 1], linestyle='-', color=colors[num],
                     label=framework[num]+'-'+method)
    ax_temp.plot(data[1:, 0], np.ones(data[1:, 0].shape), linestyle='--', color='black')
    ax_temp.set_xlabel("Significance level", fontsize=10, fontweight='bold')
    ax_temp.set_ylabel("Average size", fontsize=10, fontweight='bold')
    ax_temp.set_xlim(0, 1)
    ax_temp.set_ylim(0, 9)
    # ax_temp.set_xticklabels(fontsize=8, fontweight='bold')
    # ax_temp.set_yticklabels(fontsize=8, fontweight='bold')
    ax_temp.legend(loc='best')
    ax_temp.set_title(titls[index])

plt.show()