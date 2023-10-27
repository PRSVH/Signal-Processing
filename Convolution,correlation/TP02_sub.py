import numpy as np
import h5py
from matplotlib import pyplot as plt

def make_conv(a,b):
    # backbone of your "own" convolution product..
    # c = ...
    return #c

def make_plts(a,b,c1,c2=None,label_c='convolution'):
    # plot signal "a", "b" and the resulting 
    # convolution "c" as 3 different subplots
    # a second results "c2" can be added
    th_len_conv = len(a)+len(b)-1
    plt.figure()
    plt.subplot(311)
    plt.plot(a)
    plt.xlim([0,th_len_conv])
    plt.ylabel('signal a')
    plt.subplot(312)
    plt.plot(b,'r')
    plt.xlim([0,th_len_conv])
    plt.ylabel('signal b')
    plt.subplot(313)
    plt.plot(c1,'k')
    if c2 is not None:
        plt.plot(c2,'y--')
        plt.legend(['c1','c2'])
    plt.xlim([0,th_len_conv])
    plt.ylabel(label_c)


def read_matv73(filename):
    # input filename (mat, h5 ...) and return a dict of variables
    f=h5py.File(filename,'r')
    dic={}
    for k in f.keys():
        dic[k] = f[k][()][0,:]
    return dic
