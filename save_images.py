import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torchvision import datasets, transforms
from poison_dataset import MyDataset
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor()
])

test_data = datasets.CIFAR10(root='/home/zhuo/zqz/data', train=False, download=False, transform=transform)

percent = 5
poison_label = 8
window_size = 32
pos_list = [(31, 31), (15, 15)]
magnitude = 0.2

poison_test_dataset = MyDataset(test_data, 100, poison_label, window_size, pos_list, magnitude)

poison = False
clean_directory = './clean_images'
poison_directory = './poison_images'


def save_iamge(dataset, poison, directory):
    i = 0
    for img, label in dataset:
        img = transforms.ToPILImage()(img)
        if not poison:
            filename = f"clean_img{i}_{label}.png"
            path = os.path.join(directory, filename)
        else:
            filename = f"poison_img{i}_{label}.png"
            path = os.path.join(directory, filename)
        img.save(path)
        if i == 10:
            break
        i += 1


save_iamge(dataset=test_data, poison=False, directory=clean_directory)
save_iamge(dataset=poison_test_dataset, poison=True, directory=poison_directory)
