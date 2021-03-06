#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 下午2:19
# @Author  : LeonHardt
# @File    : parameter_tuning.py


import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""
Parameter
"""
NAME_SAMPLE = ''
DIR_SIZE = ""

if os.path.exists("./parameter_result/" + DIR_SIZE + NAME_SAMPLE) is not True:
    os.makedirs("./parameter_result/" + DIR_SIZE + NAME_SAMPLE)

if __name__ == '__main__':
    X = np.loadtxt('ginseng_x_sample.txt', delimiter=',')  # change load-date here
    y = np.loadtxt('ginseng_y_label.txt', delimiter=',', dtype='int8')

    """
    choose the classification method:
    Methods:
        1. Logistic Regression -------------------------     'lr'
        2. Support Vector Machine -------------------------  'svm'
        3. Decision Tree     ---------------------------     'tree'
        4. Random Forest    ------------------------         'forest'
    """
    # Set the parameter of the classification method
    METHOD = 'ALL'
    REPORT_NEED = 0
    param_range = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
                   5.0, 10.0, 40.0, 80.0, 100.0, 400.0, 800.0, 1000.0, 4000.0, 8000.0, 10000.0, 20000.0, 30000.0]
    # param_range = [0.1, 1.0, 10.0]
    max_depth_range = range(2, 50, 2)

    """
    if method is 'LR':
    """
    if METHOD is 'LR' or METHOD is 'ALL':
        pip_lr = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression(random_state=1))])

        param_grid = [{'clf__C': param_range}]
        gs = GridSearchCV(estimator=pip_lr, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X, y)

        # Show the full report (Yes: 1 ; No: 0)
        if REPORT_NEED is 1:
            print()
            for number in range(len(param_range)):
                print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                    gs.cv_results_['std_test_score'][number],
                                                    gs.cv_results_['params'][number],))

        with open("./parameter_result/" + DIR_SIZE + NAME_SAMPLE + "/lr_parameter.txt", "w") as lr:
            lr.write('mean_test_score: ' + str(gs.cv_results_['mean_test_score']) + '\n' + 'std_test_score: ' +
                     str(gs.cv_results_['std_test_score']) + '\nparameters: ' + str(gs.cv_results_['params']) + '\n' +
                     'Best parameters of Logistic Regression is: ' + str(gs.best_params_) + '\n' +
                     'Best score of Logistic Regression is: ' + str(gs.best_score_))
        print("Best parameters of Logistic Regression is:")
        print(gs.best_params_)
        print(gs.best_score_)

    """
    if method is 'SVM':
    """
    if METHOD is 'SVM' or METHOD is 'ALL':
        pip_svm = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
        param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
                      {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
        gs = GridSearchCV(estimator=pip_svm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X, y)

        # Show the full report (Yes: 1 ; No: 0)
        if REPORT_NEED is 1:
            print()
            for number in range(len(param_range)):
                print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                    gs.cv_results_['std_test_score'][number],
                                                    gs.cv_results_['params'][number],))
        with open(("./parameter_result/" + DIR_SIZE + NAME_SAMPLE + "/SVM_parameter.txt"), "w") as svm_parameter:
            svm_parameter.write('mean_test_score: ' + str(gs.cv_results_['mean_test_score']) + '\n' +
                                'std_test_score: ' + str(gs.cv_results_['std_test_score']) + '\nparameters: ' +
                                str(gs.cv_results_['params']) + '\n' +
                                'Best parameters of SVM is: ' + str(gs.best_params_) + '\n' +
                                'Best score of SVM is: ' + str(gs.best_score_))
        print("Best parameters of SVM is:")
        print(gs.best_params_)
        print(gs.best_score_)

    """
    if method is 'tree':
    """
    if METHOD is 'TREE' or METHOD is 'ALL':
        pip_tree = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier(random_state=1))])
        param_grid = [{'clf__criterion': ['entropy'], 'clf__max_depth': max_depth_range}]
        gs = GridSearchCV(estimator=pip_tree, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X, y)

        # Show the full report (Yes: 1 ; No: 0)
        if REPORT_NEED is 1:
            print()
            for number in range(len(param_range)):
                print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                    gs.cv_results_['std_test_score'][number],
                                                    gs.cv_results_['params'][number],))
        with open(("./parameter_result/" + DIR_SIZE + NAME_SAMPLE + "/Tree_parameter.txt"), "w") as tree_parameter:
            tree_parameter.write('mean_test_score: ' + str(gs.cv_results_['mean_test_score']) + '\n' +
                                 'std_test_score: ' + str(gs.cv_results_['std_test_score']) + '\nparameters: ' +
                                 str(gs.cv_results_['params']) +
                                 '\n' + 'Best parameters of Tree is: ' + str(gs.best_params_) + '\n' +
                                 'Best score of Tree is: ' + str(gs.best_score_))
        print("Best parameters of Tree is:")
        print(gs.best_params_)
        print(gs.best_score_)

    """
    if method is 'forest':
    """
    if METHOD is 'FOREST' or METHOD is 'ALL':
        pip_forest = Pipeline([('scl', StandardScaler()), ('clf', RandomForestClassifier(random_state=1))])
        param_grid = [{'clf__criterion': ['entropy'], 'clf__max_depth': max_depth_range}]
        gs = GridSearchCV(estimator=pip_forest, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
        gs = gs.fit(X, y)

        # Show the full report (Yes: 1 ; No: 0)
        if REPORT_NEED is 1:
            print()
            for number in range(len(param_range)):
                print("%0.3f (+/-%0.03f) for %r" % (gs.cv_results_['mean_test_score'][number],
                                                    gs.cv_results_['std_test_score'][number],
                                                    gs.cv_results_['params'][number],))
        with open(("./parameter_result/" + DIR_SIZE + NAME_SAMPLE + "/forest_parameter.txt"), "w") as forest_parameter:
            forest_parameter.write('mean_test_score: ' + str(gs.cv_results_['mean_test_score']) + '\n' +
                                   'std_test_score: ' + str(gs.cv_results_['std_test_score']) + '\nparameters: ' +
                                   str(gs.cv_results_['params']) +
                                   '\n' + 'Best parameters of Forest is: ' + str(gs.best_params_) + '\n' +
                                   'Best score of Forest is: ' + str(gs.best_score_))
        print("Best parameters of FOREST is:")
        print(gs.best_params_)
        print(gs.best_score_)
