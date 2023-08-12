# 实战-手写数字识别模型
import time
import torch
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义对数据的变换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
    # 由于该数据集只有一个通道，所以对应的均值和方差自然也只有一个
])

# 下载数据集
data_train = datasets.MNIST(root="./dataset", transform=transform, train=True, download=True)
data_test = datasets.MNIST(root="./dataset", transform=transform, train=False, download=True)

# 数据预览与数据装载
data_loader_train = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
data_loader_test = DataLoader(dataset=data_test, batch_size=64, shuffle=True)


# images, labels = next(iter(data_loader_train))
# print(images.shape)
# img = torchvision.utils.make_grid(images)
#
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img*std+mean
# print([labels[i] for i in range(64)])
# plt.imshow(img)
# plt.show()

# 模型搭建和参数优化
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x),
        x = self.dense(x[0])
        return x


# 实例化模型，选择训练设备
model = Model()
model.to(device)
# print(model)

# 定义损失函数
cost = torch.nn.CrossEntropyLoss()
cost.to(device)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters())

# 定义训练与测试步骤
total_train_step = 0
total_test_step = 0
# 这里的测试其实和epoch一样了，意义不大了

# 定义总轮数
n_epochs = 5

# 添加tensorboard
writer = SummaryWriter("logs_apply1")

# 开始跑模型
start_time = time.time()
for epoch in range(n_epochs):
    # 根据所有数据训练5轮，每轮都使用全部训练数据
    print("Epoch {}/{}".format(epoch + 1, n_epochs))
    print("-" * 10)

    # 训练开始
    model.train()
    # 定义训练过程中的预测准确的个数
    running_correct = 0
    for data in data_loader_train:
        # 每一次训练，每次取出loader中的一个batch的数据，目前设置的大小是64
        X_train, Y_train = data
        # 使用GPU训练
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)

        outputs = model(X_train)
        # 获得预测的标签->index
        _, pred = torch.max(outputs.data, 1)
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失
        # print(outputs.shape)
        # print(Y_train.shape)
        loss = cost(outputs, Y_train)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算这一次训练预测是否正确，更新预测正确的个数
        running_correct += torch.sum(pred == Y_train.data)
        # 训练次数+1
        total_train_step += 1
        if total_train_step % 100 == 0:
            # 100次训练打印一次
            end_time = time.time()
            # 打印当次训练的训练次数以及loss
            print("训练次数:{}, Train Time :{}s, Train Loss is:{}".format(total_train_step, end_time - start_time, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 这一轮的训练结束，打印训练的预测精度
    print("Train Accuracy is:{:.4f}%".format(100 * running_correct / len(data_train)))

    # 测试开始
    model.eval()
    # 定义测试过程中预测正确的个数
    testing_correct = 0
    # 定义预测过程中的总loss
    total_test_loss = 0.0
    with torch.no_grad():
        for data in data_loader_test:
            # 每一次预测取出一个batch的数据
            X_test, Y_test = data
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            outputs = model(X_test)
            loss = cost(outputs, Y_test)
            # 更新总loss
            total_test_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == Y_test.data)

    # 这一轮模型训练的测试过程结束
    total_test_step += 1
    # 打印这一轮测试的loss均值和模型在测试集上的表现
    print("Total Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(total_test_loss / len(data_test),
                                                                       100 * testing_correct / len(data_test)))
    writer.add_scalar("test_total_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", testing_correct / len(data_test), total_test_step)

    # 保存这一轮训练出的模型
    torch.save(model, "apply1_{}.pth".format(epoch + 1))
    print("模型已保存！")

writer.close()
end_time = time.time()
# 打印总耗时
print("Total Time: {}s".format(end_time - start_time))
