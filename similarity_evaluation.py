import os.path

from utils import calculate_psnr_color
from PIL import Image
import numpy as np

# img1_path = '/home/zhuo/zqz/FTROJAN/clean_images/clean_img0_3.png'
# img2_path = '/home/zhuo/zqz/FTROJAN/poison_images/poison_img0_8.png'
# img1 = Image.open(img1_path)
# img2 = Image.open(img2_path)
# img1, img2 = np.array(img1), np.array(img2)
#
# psnr = calculate_psnr_color(img1, img2)
# print(psnr)


def show_multiple_psnr(clean_folder, poison_folder):
    clean_images_paths = [os.path.join(clean_folder, img) for img in sorted(os.listdir(clean_folder)) if img.endswith('png')]
    poison_images_paths = [os.path.join(poison_folder, img) for img in sorted(os.listdir(poison_folder)) if img.endswith('png')]
    psnr_list = []
    for i in range(len(clean_images_paths)):
        img1 = Image.open(clean_images_paths[i])
        img2 = Image.open(poison_images_paths[i])

        img1, img2 = np.array(img1), np.array(img2)
        psnr = calculate_psnr_color(img1, img2)
        psnr_list.append(psnr)
    return psnr_list


clean_folder = 'clean_images'
poison_folder = 'poison_images'
psnr_list = show_multiple_psnr(clean_folder, poison_folder)
print(psnr_list)