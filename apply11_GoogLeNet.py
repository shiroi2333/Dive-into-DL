# 实战-动手学深度学习7.4节GoogLeNet
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 搭建网络
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1*1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1*1卷积层后接3*3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1*1卷积层后接5*5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3*3最大汇聚层后接1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上来凝结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


# 第一个模块：64通道，7*7卷积层
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 第二个模块：第一个卷积64通道，1*1卷积层；第二个卷积通道数增加三倍，3*3卷积层
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 第三个模块：两个完整的Inception块
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 第四个模块：五个Inception
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


# 第五个模块：两个Inception+全局平均池化
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())


net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
# print(net)

# 测试输出维度
# X = torch.rand(size=(1, 1, 96, 96))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)

# 加载数据
# 定义对数据的变换操作
transform = transforms.Compose([
    # 先对图像的尺寸进行修改，然后再转换成张量
    transforms.Resize([96, 96]),
    transforms.ToTensor()
])

# 下载数据集
data_train = datasets.FashionMNIST(root="./dataset", transform=transform, train=True, download=True)
data_test = datasets.FashionMNIST(root="./dataset", transform=transform, train=False, download=True)

# 加载数据集
batch_size = 128
data_loader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)


# 准备训练
def train_and_test(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print('training on', device)
    net.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)

    # 定义训练与测试步骤
    total_train_step = 0
    total_test_step = 0  # 这里其实和epoch是一致的

    # 定义训练损失、训练精度、测试精度的list，用于画图
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 开始计时
    start_time = time.time()
    for epoch in range(num_epochs):
        net.train()
        # 定义训练过程中的预测准确的个数
        running_correct = 0
        # 暂时的训练损失，每51个batch更新一次,每个epoch打印一次
        train_loss = 0
        for data in train_iter:
            X_train, Y_train = data
            # 使用GPU训练
            X_train = X_train.to(device)
            Y_train = Y_train.to(device)

            outputs = net(X_train)
            # 获得预测的标签->index
            _, pred = torch.max(outputs.data, 1)
            # 梯度清零
            optimizer.zero_grad()
            # 计算损失
            l = loss(outputs, Y_train)
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()

            # 计算这一次训练预测是否正确，更新预测正确的个数
            running_correct += torch.sum(pred == Y_train.data)
            # 训练次数+1
            total_train_step += 1
            if total_train_step % 51 == 0:
                # 训练阶段，每51个batch更新一遍train_loss_list
                train_loss_list.append(l.item())
                train_loss = l.item()
        # 这一轮的训练结束，打印该epoch的loss和训练的预测精度,并更新train_acc_list
        print(30 * "*")
        print("Epoch:{}".format(epoch + 1))
        print("Train Loss is:{:.4f}".format(train_loss))
        print("Train Accuracy is:{:.4f}%".format(100 * running_correct / len(data_train)))
        train_acc_list.append(float(running_correct / len(data_train)))

        net.eval()
        # 定义测试过程中预测正确的个数
        testing_correct = 0
        with torch.no_grad():
            for data in test_iter:
                # 每一次预测取出一个batch的数据
                X_test, Y_test = data
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)

                outputs = net(X_test)
                _, pred = torch.max(outputs.data, 1)
                testing_correct += torch.sum(pred == Y_test.data)

        # 这一轮模型训练的测试过程结束
        total_test_step += 1
        # 打印这一轮测试的预测精度，并更新test_acc_list
        print("Test Accuracy is:{:.4f}%".format(100 * testing_correct / len(data_test)))
        test_acc_list.append(float(testing_correct / len(data_test)))

    # 结束计时
    end_time = time.time()
    # 打印总耗时
    print(30 * "*")
    print("Total Time is:{:.4f}s".format(end_time - start_time))

    # 画图
    train_loss_x_list = list(range(1, len(train_loss_list) + 1))
    train_loss_x_list = np.divide(train_loss_x_list, len(train_loss_list) / num_epochs)
    plt.plot(train_loss_x_list, train_loss_list, color='blue', linestyle='solid', label="train loss")

    plt.plot(list(range(1, num_epochs + 1)), train_acc_list, color='red', linestyle='dashed', label="train acc")
    plt.plot(list(range(1, num_epochs + 1)), test_acc_list, color='green', linestyle='dashdot', label="test acc")
    plt.legend()
    plt.xlim([1, num_epochs])
    plt.xlabel("epoch")
    plt.show()


lr, num_epochs = 0.1, 10
train_and_test(net, data_loader_train, data_loader_test, num_epochs, lr, device)
"""
training on cuda
******************************
Epoch:1
Train Loss is:1.0007
Train Accuracy is:28.2300%
Test Accuracy is:63.9400%
******************************
Epoch:2
Train Loss is:0.5859
Train Accuracy is:70.0900%
Test Accuracy is:78.1300%
******************************
Epoch:3
Train Loss is:0.4168
Train Accuracy is:79.7250%
Test Accuracy is:81.7000%
******************************
Epoch:4
Train Loss is:0.4619
Train Accuracy is:84.6000%
Test Accuracy is:86.3900%
******************************
Epoch:5
Train Loss is:0.5118
Train Accuracy is:86.7467%
Test Accuracy is:86.9900%
******************************
Epoch:6
Train Loss is:0.2272
Train Accuracy is:88.0633%
Test Accuracy is:87.0500%
******************************
Epoch:7
Train Loss is:0.2752
Train Accuracy is:88.7233%
Test Accuracy is:86.4200%
******************************
Epoch:8
Train Loss is:0.3901
Train Accuracy is:89.5250%
Test Accuracy is:89.0100%
******************************
Epoch:9
Train Loss is:0.3223
Train Accuracy is:90.0883%
Test Accuracy is:89.5100%
******************************
Epoch:10
Train Loss is:0.2145
Train Accuracy is:90.7000%
Test Accuracy is:89.8200%
******************************
Total Time is:889.0887s
"""