import numpy as np
import torch
from utils import poison_frequency

class MyDataset(torch.utils.data.Dataset):
    # poison_label = NOne,表示i-1
    def __init__(self, dataset, percent, poison_label, window_size, pos_list, magnitude):
        self.dataset = dataset
        self.percent = percent
        self.poison_label = poison_label
        self.window_size = window_size
        self.pos_list = pos_list
        self.magnitude = magnitude
        self.poison_index = self.generate_poison_index()

    def generate_poison_index(self):
        num_poison = int(len(self.dataset) * self.percent / 100)
        poison_indices = np.random.choice(len(self.dataset), num_poison, replace=False)
        return poison_indices

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx in self.poison_index:
            # img = np.array(img).transpose((2, 0, 1))
            img = img.numpy()
            img = poison_frequency(img, self.window_size, self.pos_list, self.magnitude)
            img = torch.from_numpy(img).type(torch.float32)
            label = self.poison_label
        return img, label

    def __len__(self):
        return len(self.dataset)