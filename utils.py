import math
import os
import torch
import numpy as np
import cv2
import torch


# 输入tensor,chw, 输出np格式，chw
def dct(x_train, window_size):
    x_dct = np.zeros(x_train.shape)
    x_train = np.array(x_train)
    for c in range(x_train.shape[0]):
        for h in range(0, x_train.shape[1], window_size):
            for w in range(0, x_train.shape[2], window_size):
                sub_dct = cv2.dct(x_train[c][h:h+window_size, w:w+window_size])
                x_dct[c][h:h+window_size, w:w+window_size] = sub_dct
    return x_dct

# 输入np,chw,输出np,chw
def idct(x_train, window_size):
    x_idct = np.zeros(x_train.shape)
    for c in range(x_train.shape[0]):
        for h in range(0, x_train.shape[1], window_size):
            for w in range(0, x_train.shape[2], window_size):
                sub_idct = cv2.idct(x_train[c][h:h+window_size, w:w+window_size])
                x_idct[c][h:h+window_size, w:w+window_size] = sub_idct
    return x_idct


# 输入np, chw
def poison_frequency(x_train, window_size, pos_list, magnitude): # pos_list=[(32, 32)], magnitude=0.2
    x_train = dct(x_train, window_size)
    for c in range(x_train.shape[0]):
        for h in range(0, x_train.shape[1], window_size):
            for w in range(0, x_train.shape[2], window_size):
                for pos in pos_list:
                    x_train[c][h+pos[0]][w+pos[1]] += magnitude
    x_train = idct(x_train, window_size)

    return x_train


