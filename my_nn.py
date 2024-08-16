from utils import myAssert
import numpy as np
from MyTensor import MyTensor,Sum,Mul
#以下代码未完成
#线性算子
class Linear_Node():
    def __init__(self,weight:np.ndarray,bias:np.ndarray):
        self.weight=MyTensor(weight,requires_grad=True)
        self.bias=MyTensor(bias,requires_grad=True)
        self.sum=Sum()
        self.mul=Mul()
    def forward(self,x:MyTensor)->MyTensor:
        return self.sum.forward(self.mul.forward(self.weight,x),self.bias)

class MyLinearLayer():
    def __init__(self,fan_in:int,fan_out:int):
        NotImplementedError
        