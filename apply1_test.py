# 实战手写数字识别
import time
import torch
import torchvision.utils
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义对数据的变换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 下载数据集
data_test = datasets.MNIST(root="./dataset", transform=transform, train=False, download=True)

# 数据预览与数据装载
data_loader_test = DataLoader(dataset=data_test, batch_size=4, shuffle=True)


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


# 加载模型
model = torch.load("apply1_5.pth")
X_test, Y_test = next(iter(data_loader_test))
inputs = X_test.to(device)
pred = model(inputs)
pred = pred.argmax(1)

print("Predict Label is:", [i for i in pred.data])
print("Real Label is:", [i for i in Y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img*std+mean
plt.imshow(img)
plt.show()


