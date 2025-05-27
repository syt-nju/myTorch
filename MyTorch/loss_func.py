from MyTorch.myTensor import MyTensor,MyTensor,MatMul,Sum,Mul,Max,Exp,Log,Sub,SumUnary,Div
from MyTorch.myTensor import ComputationalGraph,op_forward

import numpy as np
class MSELoss():
    def forward(self,x:MyTensor,y:MyTensor,**kwargs)->MyTensor:
        return SumUnary.forward((x-y)*(x-y),axis=None)
    def __call__(self,x:MyTensor,y:MyTensor)->MyTensor:
        return self.forward(x,y)
