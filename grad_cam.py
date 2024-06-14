import os

import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from network import cifar_model

file_path = 'model/cifar10.pth'

network = cifar_model()
network = torch.load(file_path, map_location='cpu')
# print(network)
target_layer = [network.main[19]]
# print(target_layer)

transform = transforms.Compose([
    transforms.ToTensor()
])


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    img_np = np.array(img, dtype='uint8')
    img_np = img_np.astype(np.float32) / 255
    return img_np, input_tensor


def get_prediction(model, input_tensor):
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return prediction


clean_folder = 'clean_images'
poison_folder = 'poison_images'


def show_grad_cam(image_folder):
    image_paths = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith('png')]
    # print(image_paths)

    # 创建GradCAM对象
    cam = GradCAM(model=network, target_layers=target_layer)

    # 存储热力图和原始图像
    visualizations = []
    original_images = []
    predictions = []
    for img_path in image_paths:
        rgb_img, input_tensor = preprocess_image(img_path)
        # 获取类别
        target_class = get_prediction(network, input_tensor)
        targets = [ClassifierOutputTarget(target_class)]
        # 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        # 预处理热力图，用于展示
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        visualizations.append(visualization)
        original_images.append(rgb_img)
        predictions.append(target_class)

    # 显示热力图
    num_imgs = len(image_paths)
    fig, axs = plt.subplots(2, num_imgs, figsize=(5 * num_imgs, 10))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')

    for i in range(num_imgs):
        axs[0, i].imshow(original_images[i])
        axs[0, i].axis('off')
        # axs[0, i].set_title(predictions[i], fontsize=30)
        axs[0, i].set_title(classes[predictions[i]], fontsize=50)
        axs[1, i].imshow(visualizations[i])
        axs[1, i].axis('off')
    fig_name = './grad_cam_fig/' + image_folder + '_grad_cam'
    plt.savefig(fig_name)
    plt.tight_layout()
    plt.show()


show_grad_cam(clean_folder)
show_grad_cam(poison_folder)