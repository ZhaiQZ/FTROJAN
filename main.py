import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import poison_frequency


class cifar_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),        # 32 x 32 x 32
            nn.ELU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, (3, 3), padding=1),       # 32 x 32 x 32
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),           # 32 x 16 x 16
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, (3, 3), padding=1),       # 64 x 16 x 16
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), padding=1),       # 64 x 16 x 16
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),           # 64 x 8 x 8
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, (3, 3), padding=1),      # 128 x 8 x 8
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), padding=1),     # 128 x 8 x 8
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),            # 128 x 4 x 4
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(2048, 10)
        )

    def forward(self, x_train):
        return self.main(x_train)


transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='/home/zhuo/zqz/data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='/home/zhuo/zqz/data', train=False, download=False, transform=transform)
# print(f'Data shape:{train_data.data.shape}, sample shape:{train_data[0][0].shape}')


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


percent = 5
poison_label = 8
window_size = 32
pos_list = [(31, 31), (15, 15)]
magnitude = 0.2



batch_size = 32
num_epochs = 50
learning_rate = 0.001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
poison_train_dataset = MyDataset(train_data, percent, poison_label, window_size, pos_list, magnitude)
poison_test_dataset = MyDataset(test_data, 100, poison_label, window_size, pos_list, magnitude)
poison_train_loader = DataLoader(dataset=poison_train_dataset, batch_size=batch_size, shuffle=True)
poison_test_loader = DataLoader(dataset=poison_test_dataset, batch_size=batch_size, shuffle=False)
clean_test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

# print(poison_test_dataset[0][0].shape)

model = cifar_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

file_path = 'model/cifar10.pth'
start = time.time()
for epoch in range(num_epochs):
    model.train()
    total_loss = torch.tensor(0.0).to(device)
    correct = torch.tensor(0).to(device)
    total = torch.tensor(0).to(device)
    batch_idx = torch.tensor(0).to(device)
    for inputs, labels in poison_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        total_loss += loss
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        batch_idx += 1
    print('Training: Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%'.format(epoch, total_loss/batch_idx, 100 * correct/total))

    total_loss = 0.0
    correct = 0.0
    total = 0.0
    batch_idx = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in clean_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            total_loss += loss
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
        print('Benign Accuracy Evaluation: Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%'.format(epoch, total_loss/batch_idx, 100 * correct/total))

    total_loss = 0.0
    correct = 0.0
    total = 0.0
    batch_idx = 0
    with torch.no_grad():
        for inputs, labels in poison_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            total_loss += loss
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
        print('Attack Success Rate Evaluation: Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%\n'.format(epoch, total_loss/batch_idx, 100 * correct / total))

torch.save(model, file_path)
end = time.time()
print('using {}'.format(end-start))


































