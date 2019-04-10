#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:14:42 2019

@author: sebastien
"""

import numpy as np

def create_training(x, y):
    x1=[]
    x2=[]
    x3=[]
    
    for i in range(len(y)):
        
        if y[i]==1 or y[i]==2:
            x2.append(x[i])
            
        elif y[i]==3 or y[i]==4:
            x3.append(x[i])
            
        elif y[i]==5 or y[i]==6:
            x1.append(x[i])
            
        else:
            print("Class error at index : ", i)
    
    return(x1, x2, x3)
    
    
def average_3by3(x1, x2, x3, y):

# faut que j'affiche le nombre d'essai qui ne sont pas prient en compte 
## nb essais tot - nb essais prient dans cette fonction

    x1_mean = []
    x2_mean = []
    x3_mean = []
    y1 = []
    y2 = []
    y3 = []
    
    for i in range(len(x1)//3):
        list_av = []
        list_av.append(x1[3*i])
        list_av.append(x1[3*i+1])
        list_av.append(x1[3*i+2])
        x1_mean.append(np.mean(list_av, axis=0))
        y1.append(1)
    
    for i in range(len(x2)//3):
        list_av = []
        list_av.append(x2[3*i])
        list_av.append(x2[(3*i)+1])
        list_av.append(x2[(3*i)+2])
        x2_mean.append(np.mean(list_av, axis=0))
        y2.append(2)
    
    for i in range(len(x3)//3):
        list_av = []
        list_av.append(x3[3*i])
        list_av.append(x3[3*i+1])
        list_av.append(x3[3*i+2])
        x3_mean.append(np.mean(list_av, axis=0))
        y3.append(3)
        
    return(x1_mean+x2_mean, x2_mean+x3_mean, x1_mean+x3_mean, y1, y2, y3)
    
    
def create_training_none_av(x, y):
    x1=[]
    x2=[]
    x3=[]
    y1=[]
    y2=[]
    y3=[]
    
    for i in range(len(y)):
    
        if y[i]==1 or y[i]==2:
            y2.append(2)
            x2.append(x[i])
            
        elif y[i]==3 or y[i]==4:
            y3.append(3)
            x3.append(x[i])
            
        elif y[i]==5 or y[i]==6:
            y1.append(1)
            x1.append(x[i])
            
        else:
            print("Class error at index : ", i)
        
    return(x1+x2, x2+x3, x1+x3, y1, y2, y3)
    

def cut_time(x, file):
    t=file.times
    time_index=[]
    time=[]
    
    for i in range(len(t)):
        if -0.2 <= t[i] <= 1:
            time_index.append(i)
            time.append(t[i])
            
    new_x =  np.ndarray(shape=(x.shape[0], x.shape[1], len(time_index)))
    
    for aa in range(x.shape[0]):
        for bb in range(x.shape[1]):
            for  cc in range(len(time_index)):
                new_x[aa][bb][cc]=x[aa][bb][time_index[cc]]
    return(new_x, time)