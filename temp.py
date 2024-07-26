#临时文件，不作任何用途，核心用途可能是拿来访问torch代码
#导入 pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SimpleModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 创建输入数据和目标
input_data = torch.randn(1, 10)
target = torch.randn(1, 1)

# 前向传播
output = model(input_data)
loss = criterion(output, target)

# 反向传播
loss.backward()

# 访问和打印梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Parameter: {name}, Gradient: {param.grad}")

# 更新参数
optimizer.step()
