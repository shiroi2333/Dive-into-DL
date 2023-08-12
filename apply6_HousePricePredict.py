# Kaggle房价预测 -《动手学深度学习》
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 加载数据
train_data = pd.read_csv(r'D:\study\pytorchLearnProject\dataset\KaggleHouse\train.csv')
test_data = pd.read_csv(r'D:\study\pytorchLearnProject\dataset\KaggleHouse\test.csv')

# 检查样本和特征的数量
# print("删除Id列前训练集大小: {} ".format(train_data.shape))
# print("删除Id列前测试集大小: {} ".format(test_data.shape))
"""
删除Id列前训练集大小: (1460, 81) 
删除Id列前测试集大小: (1459, 80) 
"""

# 删掉ID列
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)
"""
(2919, 79)
"""

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理离散值---->独热编码------“Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)
"""
(2919, 330)
"""

# 获得一些数据
n_train = train_data.shape[0]
# 查找后加的一句：需要先强制将numpy中的数据转换成float
all_features = all_features.astype(float)

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
# print(train_features.shape)
# print(train_labels.shape)
# print(test_features.shape)
"""
torch.Size([1460, 330])
torch.Size([1460, 1])
torch.Size([1459, 330])
"""

# 定义损失函数
loss = nn.MSELoss()
# 获得输入特征数
in_features = train_features.shape[1]


# 一个简单的线性层
def get_net():
    net = nn.Sequential(
        # nn.Linear(in_features, 1)
        # 调参
        nn.Flatten(),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1)
    )
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


# 训练
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    # train_iter = d2l.load_array((train_features, train_labels), batch_size)
    train_data = torch.cat([train_features, train_labels], dim=1)
    train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for data in train_iter:
            X, y = data[:, :-1], data[:, -1:]
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        # if epoch % 10 == 0:
        #     print("Train Loss:{}".format(log_rmse(net, train_features, train_labels)))
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# 先不管怎么验证，先试试这个模型能不能用
# net = get_net()
# print(net)
"""
Sequential(
  (0): Linear(in_features=330, out_features=1, bias=True)
)
"""


# k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# train_ls = train(net, train_features, train_labels, test_features, num_epochs, lr, weight_decay, batch_size)


# train_data = torch.cat([train_features, train_labels], dim=1)
# print(train_data.shape)
# train_iter = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
# # print(train_iter)
# for data in train_iter:
#     print(data.shape)
#     X, y = data[:, :-1], data[:, -1:]
#     print(X.shape, y.shape)
#     print("!---!")

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 数据截取
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i in [0, 4]:
            # d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
            #          xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
            #          legend=['train', 'valid'], yscale='log')
            plt.plot(list(range(1, num_epochs + 1)), train_ls, label="train")
            plt.plot(list(range(1, num_epochs + 1)), valid_ls, label="valid")
            plt.legend()
            plt.xlim([1, num_epochs])
            plt.xlabel("epoch")
            plt.ylabel("rmse")
            plt.yscale("log")
            plt.show()
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
# 调参
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 35, 256
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
