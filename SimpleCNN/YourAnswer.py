#-*- coding: utf-8 -*-
import numpy as np
import random

from collections import OrderedDict
from utils import numerical_gradient

import torch.nn as nn


def softmax(x):
    softmax_output = None
    s = np.max(x, axis=1)[:, np.newaxis]
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)[:, np.newaxis]
    softmax_output = e_x / div
    return softmax_output


def cross_entropy_loss(score, target, thetas, lamb):
    delta = 1e-9
    batch_size = target.shape[0]

    CE_loss = 0
    reg_loss = 0

    loss = None

    CE_loss = -np.sum(target * np.log(score + delta)) / batch_size
    for value in thetas.values():
        reg_loss += np.sum(np.power(value, 2)) * lamb * 0.5
    loss = CE_loss + reg_loss
    
    return loss


class OutputLayer:

    def __init__(self, thetas, regularization):
        self.loss = None           # loss value
        self.output_softmax = None # Output of softmax
        self.target_label = None   # Target label (one-hot vector)
        self.thetas = thetas
        self.regularization = regularization
        
    def forward(self, x, y):
    
        self.output_softmax = softmax(x)
        self.target_label = y

        self.loss = cross_entropy_loss(self.output_softmax, self.target_label, self.thetas, self.regularization)

        return self.loss
    
    def backward(self, dout=1):
        
        size = self.target_label.shape[0]
        dz = None

        dz = dout * (self.output_softmax - self.target_label) / size
        
        return dz
    
    
class ReLU:

    def __init__(self):
        
        self.mask = None
        
    def forward(self, x):
        
        self.out = None
        self.out = np.maximum(0, x)
        out = self.out
    
        return out
    
    def backward(self, dout):
    
        dx = None
        self.mask = np.copy(self.out)
        self.mask[self.out > 0] = 1
        self.mask[self.out <= 0] = 0

        dx = dout * self.mask
        return dx


class Sigmoid:

    def __init__(self):
        self.out = None
        
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        out = self.out
            
        return out
    
    def backward(self, dout):
        
        dx = None
        sigmoid_prime = (1 - self.out) * self.out
        dx = dout * sigmoid_prime
        
        return dx

class Affine:
    
    def __init__(self, T, b):
        self.Theta = T
        self.b = b
        self.x = None
        self.dT = None
        self.db = None
        
    def forward(self, x):
        
        out = None
        self.x = x
        out = np.dot(self.x, self.Theta) + self.b
    
        return out
    
    def backward(self, dout):

        dx = None
        
        dx = np.dot(dout, self.Theta.T)
        self.dT = np.dot(self.x.T, dout)
        self.db = dout.mean(axis=0) * self.x.shape[0]
        
        return dx
    

from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, theta_init_std = 0.01, regularization = 0.0):

        # Weight Initialization
        self.params = {}
        self.params['T1'] = theta_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['T2'] = theta_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        self.thetas = {}
        self.thetas['T1'] = self.params['T1']
        self.thetas['T2'] = self.params['T2']
        
        self.reg = regularization

        self.layers = OrderedDict()
        
        self.layers['Affine1'] = Affine(self.thetas['T1'], self.params['b1'])
        self.layers['ReLu'] = ReLU()
        self.layers['Affine2'] = Affine(self.thetas['T2'], self.params['b2'])
        
        self.lastLayer = OutputLayer(self.thetas, self.reg)

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x

    def loss(self, x, y):
        score = self.predict(x)
        return self.lastLayer.forward(score, y)

    def accuracy(self, x, y):

        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, y):

        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['T1'] = numerical_gradient(loss_W, self.params['T1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['T2'] = numerical_gradient(loss_W, self.params['T2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        if self.reg != 0.0:
            pass
            
        return grads

    def gradient(self, x, y):

        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['T1'], grads['b1'] = self.layers['Affine1'].dT, self.layers['Affine1'].db
        grads['T2'], grads['b2'] = self.layers['Affine2'].dT, self.layers['Affine2'].db
        
        if self.reg != 0.0:
            grads['T1'] += self.reg * self.layers['Affine1'].Theta
            grads['T2'] += self.reg * self.layers['Affine2'].Theta

        return grads


class ThreeLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, theta_init_std=0.01, regularization=0.0):
        # Weight Initialization
        self.params = {}
        self.params['T1'] = theta_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['T2'] = theta_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['T3'] = theta_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.thetas = {}
        self.thetas['T1'] = self.params['T1']
        self.thetas['T2'] = self.params['T2']
        self.thetas['T3'] = self.params['T3']

        self.reg = regularization

        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.thetas['T1'], self.params['b1'])
        self.layers['ReLu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.thetas['T2'], self.params['b2'])
        self.layers['ReLu2'] = ReLU()
        self.layers['Affine3'] = Affine(self.thetas['T3'], self.params['b3'])

        self.lastLayer = OutputLayer(self.thetas, self.reg)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        score = self.predict(x)
        return self.lastLayer.forward(score, y)

    def accuracy(self, x, y):

        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy

    def gradient(self, x, y):

        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['T1'], grads['b1'] = self.layers['Affine1'].dT, self.layers['Affine1'].db
        grads['T2'], grads['b2'] = self.layers['Affine2'].dT, self.layers['Affine2'].db
        grads['T3'], grads['b3'] = self.layers['Affine3'].dT, self.layers['Affine3'].db

        if self.reg != 0.0:
            grads['T1'] += self.reg * self.layers['Affine1'].Theta
            grads['T2'] += self.reg * self.layers['Affine2'].Theta
            grads['T3'] += self.reg * self.layers['Affine3'].Theta

        return grads


class simple_CNN(nn.Module):
    def __init__(self, num_class=100):
        super(simple_CNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),

        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output


class deep_CNN(nn.Module):
    def __init__(self, num_class=100):
        super(deep_CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=num_class, bias=True),
        )

    def forward(self, x):
        output = self.features(x)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)
        return output

 
