import math
import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
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

'''
    PSNR < 20:图像质量很差
    20 < PSNR < 30:图像质量一般
    30 < PSNR <40:图像质量较好，与原始图像非常接近
    PSNR > 40 :图像质量非常好，几乎无法察觉到与原始图像的差异
'''
# 输入灰度图像
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# 输入彩色图像
def calculate_psnr_color(img1, img2):
    psnr_r = calculate_psnr(img1[:, :, 0], img2[:, :, 0])
    psnr_g = calculate_psnr(img1[:, :, 1], img2[:, :, 1])
    psnr_b = calculate_psnr(img1[:, :, 2], img2[:, :, 2])
    return (psnr_r + psnr_g + psnr_b) / 3


'''
    SSIM值位于[0, 1]，值越大，表明图像相似度越高
'''
def calculate_ssim(img1, img2):
    ssim_value, ssim_image = compare_ssim(img1, img2, full=True, channel_axis=-1)
    return ssim_value
