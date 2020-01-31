import random
import cv2 as cv
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

##########分开train.txt中的数据###############
DIRECTORY = "./DataSet/train"  # 这
# 里是自己的所有图片的位置
f = open('./DataSet/train.txt', 'r')
num_text = []  # txt中标签,012,一共5581个,str
imgs_text = []  # txt中图像名称,一共5581个,str

line = f.readline()
while line:
    a = line.split()  # 将txt分成两列
    data = a[0]  # 这是选取图像的名称，一般是xxx.jpg或其他图片格式
    imgs_text.append(data)  # 将其添加在列表之中
    label_text = a[1]  # Z这是选取图像的标签，一般是0-9的数字
    num_text.append(int(label_text))
    line = f.readline()
f.close()

def trainSetInit():
    # 产生一个0-200的随机序列。200 是自己训练图片的总张数。用于将train.txt中的数据随机排序，训练时生成随机的batch
    list = []
    for i in range(1, 5001):     # batch-size = 130
        list.append(i)
    random.shuffle(list)
    list = list[0:200]



    ##############读取图片数据######################
    labels = []  # 标签
    batch = []
    for j in range(len(list)):  # 随机取出train文件夹中的图像,200张
        num_1 = list[j]         # 第j张
        file_path = DIRECTORY + "/" + imgs_text[num_1]  # 图像的位置
        img_raw = cv.imread(file_path)  # 将图像的信息读出来
        img_resize = cv.resize(img_raw, (28, 28), interpolation=cv.INTER_AREA)  # 将图像变为指定大小
        img_singleChanel = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
        # Nomalize
        normalizedImg = np.zeros((28, 28))
        normalizedImg = cv.normalize(img_singleChanel, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)     # numpy
        normalizedImg = normalizedImg.tolist()      # 转化为list
        normalizedImg = [normalizedImg]             # 给img穿上衣服,变换维度(28,28)->(1,28,18)
        batch.append(normalizedImg)
        # cv.imshow("Image",img)
        # cv.waitKey(0)

        # transforms.Normalize((0.1307,), (0.3081,))
        labels.append(num_text[num_1])  # 标签数据存入到labels中

    labels = torch.LongTensor(labels)
    # batch = np.array(batch)
    batch = torch.Tensor(batch)
    # print("DataSet load Completely!")
    return batch, labels

def testSetInit():
    # 产生一个0-200的随机序列。200 是自己训练图片的总张数。用于将train.txt中的数据随机排序，训练时生成随机的batch
    list = []
    for i in range(5001, 5581):     # batch-size = 130
        list.append(i)

    ##############读取图片数据######################
    labels = []  # 标签
    batch = []
    for j in range(len(list)):  # 随机取出train文件夹中的图像,200张
        num_1 = list[j]         # 第j张
        file_path = DIRECTORY + "/" + imgs_text[num_1]  # 图像的位置
        img_raw = cv.imread(file_path)  # 将图像的信息读出来
        img_resize = cv.resize(img_raw, (28, 28), interpolation=cv.INTER_AREA)  # 将图像变为指定大小
        img_singleChanel = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
        # Nomalize
        normalizedImg = np.zeros((28, 28))
        normalizedImg = cv.normalize(img_singleChanel, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)     # numpy
        normalizedImg = normalizedImg.tolist()      # 转化为list
        normalizedImg = [normalizedImg]             # 给img穿上衣服,变换维度(28,28)->(1,28,18)
        batch.append(normalizedImg)
        # cv.imshow("Image",img)
        # cv.waitKey(0)

        # transforms.Normalize((0.1307,), (0.3081,))
        labels.append(num_text[num_1])  # 标签数据存入到labels中

    labels = torch.LongTensor(labels)
    # batch = np.array(batch)
    batch = torch.Tensor(batch)
    # print("DataSet load Completely!")
    return batch, labels


###########################神经网络############################
learning_rate = 0.01
epochs = 40

w1, b1 = torch.randn(200, 784, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
w2, b2 = torch.randn(200, 200, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(3, 200, requires_grad=True),\
         torch.zeros(3, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x


optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()

class monitor():
    batch_idx = 0 # 记录一个epoch中,训练到了哪个batch
    def rescount(self):
        self.batch_idx = 0

def nn():
    for i in range(30):
        #############train#################
        data, target = trainSetInit()
        data = data.view(-1, 28*28)
        logits = forward(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()
        monitor.batch_idx = monitor.batch_idx + 1
        if monitor.batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, monitor.batch_idx * 200, 6000,
                       100. * monitor.batch_idx / 30, loss.item()))


        ###############text###############
    data, target = testSetInit()
    test_loss = 0
    correct = 0
    # start = time.clock()

    data = data.view(-1, 28 * 28)
    logits = forward(data)
    test_loss += criteon(logits, target).item()     # 581样本累计loss

    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()

    test_loss /= 581
    test_loss = test_loss * 200
    # 平均loss
    # show_parameter()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 581,
        100. * correct / 581))



if __name__ == "__main__":
    for epoch in range(epochs):     # 对于十个epochs
        nn()
        monitor.batch_idx = 0





