# 实战-动手学深度学习7.3节NiN
import time
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 定义NiN块
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


# 搭建NiN模型
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转化为二维的输出，其形状为（批量大小， 10）
    nn.Flatten()
)


# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)



# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                print("Current Loss is:{:.4f}".format(train_loss))
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