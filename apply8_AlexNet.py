# 实战-动手学深度学习7.1节AlexNet
import time
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 搭建网络结构
net = nn.Sequential(
    # 使用11*11的窗口来捕捉对象
    # stride设置为4，减少输出的高度和宽度
    # 输出通道比较大
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，padding设置为2，增大通道
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 三个连续的卷积层以及较小的卷积窗口
    # 进一步轻微增加输出通道
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 使用Dropout防止过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 输出类别为10
    nn.Linear(4096, 10)
)

# 测试网络结构
# X = torch.randn(1, 1, 224, 224)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)


# 定义对数据的变换操作
transform = transforms.Compose([
    # 先对图像的尺寸进行修改，然后再转换成张量
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

# 下载数据集
data_train = datasets.FashionMNIST(root="./dataset", transform=transform, train=True, download=True)
data_test = datasets.FashionMNIST(root="./dataset", transform=transform, train=False, download=True)

# 加载数据集
batch_size = 128
data_loader_train = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_test = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)


# 打算直接套用LeNet搭建的训练测试结构
def train_and_test(net, train_iter, test_iter, num_epochs, lr, device):
    """
    :param net:
    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :param lr:
    :param device:
    :return:
    """

    """
     我们使用在 4.8.2.2节中介绍的Xavier随机初始化模型参数。 与全连接层一样，我们使用交叉熵损失函数和小批量随机梯度下降。
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    print('training on', device)
    net.to(device)

    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # 印象里，老师的教程里面好像没有把损失函数放在GPU中，不知道这句是不是必要的？
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
    # print(train_loss_list)
    # print(train_acc_list)
    # print(test_acc_list)

    # 结束计时
    end_time = time.time()
    # 打印总耗时
    print(30 * "*")
    print("Total Time is:{:.4f}s".format(end_time - start_time))

    # 画图
    # 这里一定注意，把不同量级的数据画在同一个图上的方式
    # 错误方式！！！
    # plt.plot(list(range(1, len(train_loss_list) + 1)), train_loss_list, color='blue', linestyle='solid', label="train loss")
    # 正确方式！！！
    train_loss_x_list = list(range(1, len(train_loss_list) + 1))
    train_loss_x_list = np.divide(train_loss_x_list, len(train_loss_list) / num_epochs)
    plt.plot(train_loss_x_list, train_loss_list, color='blue', linestyle='solid', label="train loss")

    plt.plot(list(range(1, num_epochs + 1)), train_acc_list, color='red', linestyle='dashed', label="train acc")
    plt.plot(list(range(1, num_epochs + 1)), test_acc_list, color='green', linestyle='dashdot', label="test acc")
    plt.legend()
    plt.xlim([1, num_epochs])
    plt.xlabel("epoch")
    plt.show()


lr, num_epochs = 0.01, 10
train_and_test(net, data_loader_train, data_loader_test, num_epochs, lr, device)
"""
Epoch:10
Train Loss is:0.3497
Train Accuracy is:88.0017%
Test Accuracy is:88.2600%
******************************
Total Time is:1043.7961s
"""
# 为啥loss还是这么抖呢？？？
