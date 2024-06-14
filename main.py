import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import poison_frequency
from network import cifar_model
from poison_dataset import MyDataset
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root='/home/zhuo/zqz/data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='/home/zhuo/zqz/data', train=False, download=False, transform=transform)
# print(f'Data shape:{train_data.data.shape}, sample shape:{train_data[0][0].shape}')

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
training_acc = []
benign_test_acc = []
attack_success_rate = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_idx = 0
    for inputs, labels in poison_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels).to(device)
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        total_loss += loss.item()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        batch_idx += 1
    print('Training: Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%'.format(epoch, total_loss/batch_idx, 100 * correct/total))
    training_acc.append(100 * correct/total)

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
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
        print('Benign Accuracy Evaluation: Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%'.format(epoch, total_loss/batch_idx, 100 * correct/total))
        benign_test_acc.append(100 * correct/total)

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
            total_loss += loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            batch_idx += 1
        print('Attack Success Rate Evaluation: Epoch: {}, Loss: {:.4}, Accuracy: {:.4}%\n'.format(epoch, total_loss/batch_idx, 100 * correct / total))
        attack_success_rate.append(100 * correct / total)


torch.save(model, file_path)
end = time.time()
print('using {} seconds'.format(end-start))

# 训练准确率，干净测试准确率，攻击成功率曲线
plt.figure(figsize=(10, 6))
x = range(len(training_acc))
plt.plot(x, training_acc, label='training accuracy', marker='o')
plt.plot(x, benign_test_acc, label='benign test ccuracy', marker='s')
plt.plot(x, attack_success_rate, label='attack success rate', marker='^')
plt.xlabel('epoch')
plt.ylabel('value %')
plt.legend()
plt.grid(True)
figname = './fig.png'
plt.savefig(figname)































