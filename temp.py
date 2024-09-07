import MyTensor
import my_nn
import numpy as np
from torch.optim import SGD
#构建一个y=ax+b的线性模型
model=my_nn.MyLinearLayer(2,1,initial_policy='zeros')
#创建一个形状为2，2的全1张量
x=MyTensor.MyTensor(np.ones((2,2)),requires_grad=False)
y=model.forward(x)
y.backward()
print(model.weight.grad)
