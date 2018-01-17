# -*- coding: utf-8 -*-
# @Time    : 2018/1/17 14:49
# @Author  : LeonHardt
# @File    : significance_error_knn.py


# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 9:53
# @Author  : LeonHardt
# @File    : bcp_classification.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from nonconformist.base import ClassifierAdapter
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory, ClassifierNc
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.acp import BootstrapConformalClassifier

# ----------------------------------------
# preprocessing
# -----------------------------------------
path = os.getcwd()

X = np.loadtxt('ginseng_x_sample.txt', delimiter=',')
y = np.loadtxt('ginseng_y_label.txt', delimiter=',')

sc = StandardScaler()
X = sc.fit_transform(X)

# --------------------------------------------
# prediction
# --------------------------------------------

summary = []


# simple_model = KNeighborsClassifier(n_neighbors=3)
# model_name = '3NN'

simple_model = KNeighborsClassifier(n_neighbors=1)
model_name = '1NN'

# ------------------------------------------------------------------------------
# prediction with significance
framework_name = 'CP'
error_summary = []
for sig in np.arange(0.1, 1.0, 0.002):
    print('sig = ' + str(sig))
    s_folder = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for k, (train, test) in enumerate(s_folder.split(X, y)):
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        truth = y_test.reshape((-1, 1))
        # -----------------------------------------------
        # BCP
        # conformal_model = BootstrapConformalClassifier(IcpClassifier(ClassifierNc(ClassifierAdapter(simple_model))))
        # conformal_model.fit(x_train_sp, y_train_sp)

        # ------------------------------------------
        # ICP
        # x_train_sp, x_cal, y_train_sp, y_cal = train_test_split(x_train, y_train, test_size=0.3, shuffle=True,
        #                                                         random_state=1)
        # nc = NcFactory.create_nc(model=simple_model)
        # conformal_model = IcpClassifier(nc)
        # conformal_model.fit(x_train_sp, y_train_sp)
        # conformal_model.calibrate(x_cal, y_cal)

        # ---------------------------------------------------
        # CP
        nc = NcFactory.create_nc(model=simple_model)
        conformal_model = IcpClassifier(nc)
        conformal_model.fit(x_train, y_train)
        conformal_model.calibrate(x_train, y_train)

        prediction = conformal_model.predict(x_test, significance=None)
        table = np.hstack((prediction, truth))
        result = [class_mean_errors(prediction, truth, significance=sig),
                  class_avg_c(prediction, truth, significance=sig)]
        if k == 0:
            summary = result
        else:
            summary = np.vstack((summary, result))
        # print('\nBCP')
        # print('Accuracy: {}'.format(result[0]))
        # print('Average count: {}'.format(result[1]))

    df_summary = pd.DataFrame(summary, columns=['Accuracy', 'Average_count'])
    temp = [sig, df_summary['Accuracy'].mean()]

    if sig == 0.1:
        error_summary = temp
        print(error_summary)
        print(len(error_summary))
    else:
        error_summary = np.vstack((error_summary, temp))

save_path = os.getcwd()+'/summary/' + framework_name+'/'
if os.path.exists(save_path) is not True:
    os.makedirs(save_path)

save_file = save_path + 'significance_error_'+model_name +'_'+framework_name+'.txt'
if os.path.exists(save_file):
    os.remove(save_file)
np.savetxt(save_file, error_summary, delimiter=',')
    # print(df_summary)
    # print(df_summary['Accuracy'].mean())
    # print(type(df_summary['Accuracy'].mean()))
#
# summary_path = path + '/summary/bcp/'
# summary_file = summary_path + 'bcp_' + model_name + '.csv'
# if os.path.exists(summary_path) is not True:
#     os.makedirs(summary_path)
# if os.path.exists(summary_file) is True:
#     os.remove(summary_file)


# ------------------------------------------------------------------------------------
# force_prediction
# result_summary = []
# s_folder = StratifiedKFold(n_splits=10, shuffle=True)
# for index, (train, test) in enumerate(s_folder.split(X, y)):
#     x_train, x_test = X[train], X[test]
#     y_train, y_test = y[train], y[test]
#     truth = y_test.reshape((-1, 1))
#
#     model = BootstrapConformalClassifier(IcpClassifier(ClassifierNc(ClassifierAdapter(simple_model))))
#     model.fit(x_train, y_train)
#     prediction = model.predict(x_test, significance=None)
#     table = np.hstack((prediction, truth))
#     result = [1 - force_mean_errors(prediction, truth)]
#
#     if index == 0:
#         result_summary = result
#     else:
#         result_summary = np.vstack((result_summary, result))
#     print('\nBCP_Force')
#     if np.unique(y_test).shape[0] == 10:
#         print('True')
#     else:
#         print('Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#     print('Accuracy: {}'.format(result[0]))
#
# df_summary = pd.DataFrame(result_summary, columns=['Accuracy'])
# print(df_summary)
# summary_path = path + '/summary/bcp/force/'
# print(df_summary.mean())
# summary_file = summary_path + 'bcp_' + model_name + '.csv'
# if os.path.exists(summary_path) is not True:
#     os.makedirs(summary_path)
# if os.path.exists(summary_file) is True:
#     os.remove(summary_file)
#
#
# df_summary.to_csv(summary_file)






