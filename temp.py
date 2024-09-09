import numpy as np
from torch import nn,tensor,float32
from torch.optim import Adagrad
from torch.nn import MSELoss
#构造y=3x+2
x=np.array(range(10)).reshape(-1,1)
x=tensor(x,dtype=float32)
y_true=3*x+2+np.random.randn(10,1)*0.001
y_true=tensor(y_true,dtype=float32)
model=nn.Linear(1,1)
optimizer=Adagrad(model.parameters())
loss_func=MSELoss()
for i in range(100000):
    optimizer.zero_grad()
    y_pred=model(x)
    loss=loss_func(y_pred,y_true)
    loss.backward()
    optimizer.step()
    if loss.item()<1:
        print("epcho",i)
        break
print("weight",model.weight.data,"bias",model.bias.data)
