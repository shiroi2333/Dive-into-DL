# 实战-动手学深度学习7.2节VGG
import time
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 定义VGG块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 定义VGGNet
def vgg(conv_arch):
    global out_channels
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        # 千万注意，如果报错，优先看看是不是自己哪里名字写错了，我这里就把conv_arch写成了conv_blks，因此其实根本没有卷积层了，找了半天才发现
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


# net = vgg(conv_arch)
# print(net)

# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)

# 稍作修改，构建一个通道数较小的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
# print(net)

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


# 依然重复利用训练测试结构
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


lr, num_epochs = 0.05, 10
train_and_test(net, data_loader_train, data_loader_test, num_epochs, lr, device)
"""
training on cuda
Current Loss is:2.2933
Current Loss is:2.3048
Current Loss is:2.1893
Current Loss is:1.5383
Current Loss is:0.7466
Current Loss is:0.8008
Current Loss is:0.5867
Current Loss is:0.7347
Current Loss is:0.5641
******************************
Epoch:1
Train Loss is:0.5641
Train Accuracy is:51.1300%
Test Accuracy is:81.8600%
Current Loss is:0.5483
Current Loss is:0.4341
Current Loss is:0.4909
Current Loss is:0.4350
Current Loss is:0.4746
Current Loss is:0.3580
Current Loss is:0.3677
Current Loss is:0.3009
Current Loss is:0.3670
******************************
Epoch:2
Train Loss is:0.3670
Train Accuracy is:84.1100%
Test Accuracy is:86.7500%
Current Loss is:0.3366
Current Loss is:0.3016
Current Loss is:0.2075
Current Loss is:0.2721
Current Loss is:0.3772
Current Loss is:0.2861
Current Loss is:0.2210
Current Loss is:0.2998
Current Loss is:0.2559
******************************
Epoch:3
Train Loss is:0.2559
Train Accuracy is:87.5183%
Test Accuracy is:87.7500%
Current Loss is:0.3837
Current Loss is:0.3279
Current Loss is:0.3880
Current Loss is:0.2636
Current Loss is:0.3322
Current Loss is:0.2967
Current Loss is:0.3226
Current Loss is:0.3053
Current Loss is:0.2532
******************************
Epoch:4
Train Loss is:0.2532
Train Accuracy is:89.1983%
Test Accuracy is:89.2600%
Current Loss is:0.2240
Current Loss is:0.4162
Current Loss is:0.3427
Current Loss is:0.1658
Current Loss is:0.1833
Current Loss is:0.2109
Current Loss is:0.2296
Current Loss is:0.2666
Current Loss is:0.2325
******************************
Epoch:5
Train Loss is:0.2325
Train Accuracy is:90.0967%
Test Accuracy is:90.2200%
Current Loss is:0.2075
Current Loss is:0.1895
Current Loss is:0.1845
Current Loss is:0.2435
Current Loss is:0.2265
Current Loss is:0.3574
Current Loss is:0.2299
Current Loss is:0.2525
Current Loss is:0.2970
Current Loss is:0.3314
******************************
Epoch:6
Train Loss is:0.3314
Train Accuracy is:90.8933%
Test Accuracy is:90.5600%
Current Loss is:0.2067
Current Loss is:0.2570
Current Loss is:0.2727
Current Loss is:0.2654
Current Loss is:0.2485
Current Loss is:0.2132
Current Loss is:0.3252
Current Loss is:0.2701
Current Loss is:0.1982
******************************
Epoch:7
Train Loss is:0.1982
Train Accuracy is:91.5400%
Test Accuracy is:91.0100%
Current Loss is:0.1154
Current Loss is:0.1507
Current Loss is:0.1696
Current Loss is:0.2067
Current Loss is:0.1715
Current Loss is:0.1581
Current Loss is:0.1890
Current Loss is:0.2584
Current Loss is:0.1947
******************************
Epoch:8
Train Loss is:0.1947
Train Accuracy is:92.0900%
Test Accuracy is:91.6100%
Current Loss is:0.2555
Current Loss is:0.1692
Current Loss is:0.1775
Current Loss is:0.1733
Current Loss is:0.2228
Current Loss is:0.1867
Current Loss is:0.2070
Current Loss is:0.1833
Current Loss is:0.3612
******************************
Epoch:9
Train Loss is:0.3612
Train Accuracy is:92.6183%
Test Accuracy is:91.5000%
Current Loss is:0.2076
Current Loss is:0.1644
Current Loss is:0.1979
Current Loss is:0.1589
Current Loss is:0.1443
Current Loss is:0.2457
Current Loss is:0.1340
Current Loss is:0.1849
Current Loss is:0.3659
******************************
Epoch:10
Train Loss is:0.3659
Train Accuracy is:93.1150%
Test Accuracy is:91.9700%
******************************
Total Time is:1701.2362s
"""