#基于mlp进行几个优化器的比较

from ..MyTensor import MyTensor
import numpy as np
from optim import BGD,AdaGrad,Adam,BaseOptimizer
from my_nn import MLP
from loss_func import MSELoss
from utils import random_perturbation
# 生成数据
#尝试模拟 y_1=x_1^2+2*x_1+1+3x_2+4
#假设采样充分且等距
X=np.array([i for i in range(-100,100)]).reshape(-1,2)
# x=MyTensor(x)
# y_true=x**2+2*x+1+np.random.randn(200)*100

# model=MLP(1,10,1,initial_policy='random')
# loss_func=MSELoss()
# def train_epcho(optimizer:BaseOptimizer):
#     for i in range(10000000):
#         y_pred=model.forward(MyTensor)