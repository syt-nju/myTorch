import __init__
from MyTorch import MyTensor
from MyTorch.my_nn import MLP
from MyTorch.optim import BGD
from MyTorch.loss_func import MSELoss
from MyTorch.Dataloader import mydataLoader
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


from torchvision import datasets, transforms
import os

data_dir = "dataset"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_transforms = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,)),  # 归一化，均值为0.5，标准差为0.5
])

# MNIST dataset
train_data = datasets.MNIST(data_dir, train=True, download=True, transform=data_transforms)
test_data = datasets.MNIST(data_dir, train=False, download=True, transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
print(example_targets)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Ground Truth: {}".format(example_targets[i]))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128) 
        self.fc2 = nn.Linear(128, 64)      
        self.fc3 = nn.Linear(64, 10)        
    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

criterion = nn.CrossEntropyLoss()
model_torch = Net()
optimizer = optim.SGD(model_torch.parameters(), lr=0.01)
myModel = MLP(784, 100, 10, initial_policy='zeros')
myOptimizer = BGD(myModel.parameters, lr=0.01)


def train_mytensor(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs = MyTensor.MyTensor(inputs.numpy().reshape(inputs.shape[0], -1))

        # 实际上应该用交叉熵损失函数，由于还没实现，这里用 MSE 代替，所以这里的处理有点唐，问GPT的
        target_one_hot = np.zeros((target.size(0), 10))  
        target_one_hot[np.arange(target.size(0)), target.numpy()] = 1  # 将目标转换为 one-hot 编码
        target = MyTensor.MyTensor(target_one_hot)  # 转换为自定义张量 MyTensor 格式

        myOptimizer.zero_grad()
        outputs = myModel.forward(inputs)
        loss = MSELoss().forward(outputs, target)
        loss.backward()
        myOptimizer.step()
        running_loss += loss.data
        _, predicted = torch.max(torch.tensor(outputs.data), dim=1)
        target_np = target.data
        target_labels = np.argmax(target_np, axis=1)  
        target_labels_tensor = torch.tensor(target_labels)

        running_total += inputs.shape[0]
        running_correct += (predicted == target_labels_tensor).sum().item()

        if batch_idx % 100 == 0:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch, batch_idx + 1, running_loss / 100, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0

def train_torch(epoch):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        optimizer.zero_grad()
        outputs = model_torch(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 100 == 0:
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch, batch_idx + 1, running_loss / 100, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0

if __name__ == "__main__":

    train_torch(1)
    train_mytensor(1)

    # 很简单的测试，分别用 MyTensor 和 pytorch 跑了一个训练过程
    X_train = MyTensor.MyTensor(np.array([[0,0],[0,1],[1,0],[1,1]]), requires_grad=True)
    y_train = MyTensor.MyTensor(np.array([[0],[1],[1],[0]]), requires_grad=True)

    X_torch = torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32)
    y_torch = torch.tensor([[0],[1],[1],[0]],dtype=torch.float32)
    model = torch.nn.Linear(2,1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    for epoch in range(10000):
        y_pred = model(X_torch)
        loss = loss_fn(y_pred, y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch%1000 == 0:
            print(f'epoch: {epoch}, loss: {loss.item()}')

    DataLoader = mydataLoader(X_train, y_train, 2, shuffle=True, drop_last=False)
    model = MLP(2,3,1,initial_policy='zeros')
    optimizer = BGD(model.parameters,lr=0.01)
    for epoch in range(10000):
        for i, (X_batch, y_batch) in enumerate(DataLoader):
            y_pred = model.forward(MyTensor.MyTensor(X_batch))
            loss = MSELoss().forward(y_pred, MyTensor.MyTensor(y_batch))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch%1000 == 0:
                print(f'epoch: {epoch}, batch: {i}, loss: {loss.data}')

