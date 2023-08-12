# 实战-猫狗分类-迁移VGG16
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings('ignore')

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "D:\study\pytorchLearnProject\dataset\DogsVSCats"

# 以下方法均使用了字典格式，更加方便
data_transform = {x: transforms.Compose([transforms.Resize([64, 64]),
                                         transforms.ToTensor()])
                  for x in ["train", "valid"]}

image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "valid"]}

dataloader = {x: DataLoader(dataset=image_datasets[x], batch_size=16, shuffle=True) for x in ["train", "valid"]}

# 迁移训练好的模型
model = models.vgg16(pretrained=True)

# print(model)

# 对迁移的模型进行调整 - 冻结全连接层之前的全部层次，让其的参数不会更新，更新的只要有全连接层中的参数
for parma in model.parameters():
    parma.requires_grad = False # 将参数中的该属性全部False，即冻结

# 重新定义Fc层
model.classifier = torch.nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 2)
)

model.to(device)

cost = nn.CrossEntropyLoss()
cost.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# print(model)

# 训练 & 测试
epoch_n = 5 # 训练5次
time_open = time.time()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch, epoch_n-1))
    print("-"*10)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)

        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)

            optimizer.zero_grad()

            loss = cost(y_pred, y)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)

            if batch % 500 == 0 and phase == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(batch, running_loss/batch,
                                                                             100*running_corrects/(16*batch)))

        epoch_loss = running_loss*16/len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])

        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

    time_end = time.time()
    print(time_end - time_open)

