#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 03:01:16 2020

@author: smruti
"""

import torch
import torchvision
import torchvision.transforms as transforms



def data(root_train,root_test):
   tranform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

   trainset = torchvision.datasets.ImageFolder(root_train, transform = transform)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

   testset = torchvision.datasets.ImageFolder(root_test, transform = transform)
   testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4) 
   return(trainloader,testloader)