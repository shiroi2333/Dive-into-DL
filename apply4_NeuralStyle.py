# 实战-图像风格迁移
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy

import warnings

warnings.filterwarnings('ignore')

transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor()])


def loading(path=None):
    img = Image.open(path).convert('RGB')  # 如果图像不是三通道，需要在这里进行一下修改
    img = transform(img)
    img = img.unsqueeze(0)
    return img


content_img = loading("images/content.jpg")
content_img = Variable(content_img).cuda()
style_img = loading("images/style.jpg")
style_img = Variable(style_img).cuda()


# print(content_img)
# print(style_img)

class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_fn(input * self.weight, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


class Gram_matrix(torch.nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a * b * c * d)


class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = Gram_matrix()

    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)
        self.loss = self.loss_fn(self.Gram, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


use_gpu = torch.cuda.is_available()
cnn = models.vgg16(pretrained=True).features  # 迁移VGG16架构的特征提取部分

if use_gpu:
    cnn = cnn.cuda()

model = copy.deepcopy(cnn)

# 指定整个卷积过程中分别在那一层提取内容和风格
content_layer = ["Conv_3"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]

# 定义保存内容损失和风格损失的列表
content_losses = []
style_losses = []

# 指定内容损失和风格损失对最后得到的融合图片的影响权重
content_weight = 1
style_weight = 1000

new_model = torch.nn.Sequential()

gram = Gram_matrix()

if use_gpu:
    model = model.cuda()
    new_model = new_model.cuda()
    gram = gram.cuda()

index = 1
for layer in list(model)[:8]:
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_" + str(index)
        new_model.add_module(name, layer)
        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = Content_loss(content_weight, target)
            new_model.add_module("content_loss_" + str(index), content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module("style_loss_" + str(index), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = "Relu_" + str(index)
        new_model.add_module(name, layer)
        index += 1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = "MaxPool_" + str(index)
        new_model.add_module(name, layer)

# print(new_model)
# print(model)
# print(cnn)
# print(content_losses, style_losses)

input_img = content_img.clone()
parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])

epoch_n = 300

epoch = [0]
while epoch[0] <= epoch_n:

    def closure():
        optimizer.zero_grad()
        style_score = 0
        content_score = 0
        parameter.data.clamp_(0, 1)
        new_model(parameter)
        for sl in style_losses:
            style_score += sl.backward()

        for cl in content_losses:
            content_score += cl.backward()

        epoch[0] += 1
        if epoch[0] % 50 == 0:
            print('Epoch:{} Style Loss:{:4f} Content Loss{:4f}'.format(epoch[0],
                                                                       style_score.item(), content_score.item()))
        return style_score + content_score


    optimizer.step(closure)

# 对风格迁移图片输出
output = parameter.data
unloader = transforms.ToPILImage()

plt.ion()
plt.figure()


def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, 224, 224)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause方便图像更新


imshow(output, title='Output Image')

# 设置sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
