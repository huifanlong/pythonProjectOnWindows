# import torch.nn as nn
# import torch
#
# rnn = nn.RNN(input_size=10, hidden_size=128, num_layers=1)
# input = torch.randn(5, 3, 10)  # sequence length，batch size，input size
# h0 = torch.randn(1, 3, 128)  # (D∗num_layers,batch size,output size)
# output, hn = rnn(input, h0)
# print("ok")

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from fgp_rnn_features import data, label


# 设置超参数
learning_rate = 0.0001
num_epochs = 700
batch_size = 4

# 定义输入和输出
input_dim = 12  # 输入特征数
time_steps = 40  # 时间步数
num_classes = 2  # 分类数

# data_normalized = np.where(data.std(axis=0) == 0, data, (data - np.mean(data, axis=0)) / np.std(data, axis=0))
dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.load("../result_data/data_week.npy")).to(torch.float32), torch.from_numpy(np.load("../result_data/label_week.npy")))

# 随机划分数据集
train_val_data, test_data = train_test_split(dataset, test_size=0.15, random_state=1)
train_data, val_data = train_test_split(train_val_data, test_size=0.15, random_state=1)

# 创建采样器
train_sampler = SubsetRandomSampler(range(len(train_data)))
val_sampler = SubsetRandomSampler(range(len(val_data)))
test_sampler = SubsetRandomSampler(range(len(test_data)))

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=128, sampler=train_sampler)
val_loader = DataLoader(val_data, batch_size=128, sampler=val_sampler)
test_loader = DataLoader(test_data, batch_size=128, sampler=test_sampler)

# X_train = torch.randn(100, 13, 8)  # 训练数据，形状为(num_train, time_steps, input_dim) batch_first= TRUE
# Y_train = torch.randint(2, size=(100, 2))  # 训练标签，形状为(num_train, num_classes)
# X_val = torch.randn(20, 13, 8)  # 验证数据，形状为(num_val, time_steps, input_dim)
# Y_val = torch.randint(2, size=(20, 2))  # 验证标签，形状为(num_val, num_classes)
# X_test = torch.randn(20, 13, 8)  # 测试数据，形状为(num_test, time_steps, input_dim)
# Y_test = torch.randint(2, size=(20, 2))  # 测试标签，形状为(num_test, num_classes)

# # 定义数据集和数据加载器
# train_dataset = TensorDataset(X_train, Y_train)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataset = TensorDataset(X_val, Y_val)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = TensorDataset(X_test, Y_test)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_dim, 32, 2, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, torch.argmax(batch_y, dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            train_acc += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(batch_y, dim=1))
        train_acc = train_acc.item() / len(train_data)
    val_acc = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            val_acc += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(batch_y, dim=1))
        val_acc = val_acc.item() / len(val_data)
    print("Epoch:", (epoch + 1), "loss =", "{:.3f}".format(avg_loss), "train accuracy =", "{:.3f}".format(train_acc),
          "validation accuracy =", "{:.3f}".format(val_acc))

# 测试模型
test_acc = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x)
        test_acc += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(batch_y, dim=1))
    test_acc = test_acc.item() / len(test_data)
print("Test accuracy:", test_acc)  # 0.8743
