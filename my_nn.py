from utils import myAssert
import numpy as np
from MyTensor import MyTensor,Sum,Mul,MatMul

class MyLinearLayer():
    def __init__(self,fan_in:int,fan_out:int,initial_policy:str = 'random') -> None:
        if initial_policy == 'random':
            self.weight = MyTensor(np.random.randn(fan_in,fan_out),requires_grad=True)
            self.bias = MyTensor(np.random.randn(fan_out),requires_grad=True)
    def forward(self,x:MyTensor)->MyTensor:
        #检查形状
        myAssert(x.shape[1] == self.weight.shape[0],'shape not match')
        
        matmul=MatMul()
        matmul.forward(x,self.weight)
        add=Sum()
        add.forward(matmul.output,self.bias)
        return add.output
    
if __name__ == "__main__":
    #测试MyLinearLayer
    layer = MyLinearLayer(3,2)
    x = MyTensor(np.random.randn(2,3),requires_grad=False)
    y = layer.forward(x)
    y.backward()
    print(layer.weight.grad)
    print(x.data)
    print(layer.bias.grad)
    
        
        
        