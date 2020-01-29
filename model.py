#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:48:57 2020

@author: smruti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self,hidden1=6,hidden2=16,in_feature=3,out_feature=2):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_feature,hidden1,5)
        self.pad=nn.ZeroPad2d(2)
        self.relu = nn.ReLU()
        self.pool=nn.MaxPool2d(kernal_size=2,stride=2,padding=0)
        self.conv2= nn.Conv2d(hidden1,hidden2,5)
        self.fc1 = nn.Linear(hidden2*300*300,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_feature)
 
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(self.pad(x))))
        x=self.pool(self.relu(self.conv2(self.pad(x))))
        x = x.view(-1, self.hidden2 * 300 * 300)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x