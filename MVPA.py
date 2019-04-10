#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:34:12 2019

@author: sebastien
"""
from mne.io import read_epochs_eeglab
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore, Vectorizer, Scaler, CSP, get_coef, LinearModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from function import create_training, average_3by3, create_training_none_av, cut_time

#_________________________data import______________________________#

file = read_epochs_eeglab(input_fname = "data/Subject_03_Day_02_EEG_RAW_RS_BP_EP_BL_ICA_RJ_Components_Trial.set")

dim1 = file.get_data().shape[0]
dim2 = file.get_data().shape[1]
dim3 = file.get_data().shape[2]

x = file.get_data()
y = file.events[:,2]

x, time = cut_time(x, file)


#_______________________create training data______________________#

x1, x2, x3 = create_training(x, y)

#_________________________moyenne 3 essai à 3 sur chaque channel__________________________#
            
x12, x23, x13, y1, y2, y3 = average_3by3(x1, x2, x3, y)


#________________________transform into np.ndarray________________________#

x12 = np.ndarray(shape= (len(x12), len(x12[0]), len(x12[0][0])), buffer=np.array(x12))
#x23 = np.ndarray(shape= (len(23), len(x23[0]), len(x23[0][0]), buffer=np.array(x23))
x13 = np.ndarray(shape= (len(x13), len(x13[0]), len(x13[0][0])), buffer=np.array(x13))
y12 = np.ndarray(shape= (len(y1+y2)), buffer=np.array(y1+y2), dtype=int)
#y23 = np.ndarray(shape= (len(y2+y3)), buffer=np.array(y2+y3), dtype=int)
y13 = np.ndarray(shape= (len(y3+y1)), buffer=np.array(y3+y1), dtype=int)


#_______________________________CSP filtering_________________________#

x12 = CSP(n_components=6, transform_into='csp_space').fit_transform(x12, y12)
x13 = CSP(n_components=6, transform_into='csp_space').fit_transform(x13, y13)


#_______________________________Learning________________________________#

clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs'))

time_decod = SlidingEstimator(clf, n_jobs=1, scoring=make_scorer(accuracy_score), verbose=True)
#time_decod = GeneralizingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)

scores12 = cross_val_multiscore(time_decod, x12, y12, cv=10, n_jobs=1)
#scores23 = cross_val_multiscore(time_decod, x23, y23, cv=10, n_jobs=1)
scores13 = cross_val_multiscore(time_decod, x13, y13, cv=10,  n_jobs=1)

scores12 = np.mean(scores12, axis=0)
#scores23 = np.mean(scores23, axis=0)
scores13 = np.mean(scores13, axis=0)

print(np.mean(scores12))
print(np.mean(scores13))


#_____________________________Ploting_______________________________#

fig, ax = plt.subplots()
ax.plot(time, scores12, label='score12')
#ax.plot(file.times, scores23, label='score')
ax.plot(time, scores13, label='score13')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')



