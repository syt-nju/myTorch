from utils.utils import *
from MyTensor import MyTensor,MatMul,Sum
import numpy as np
class MyLinearLayer():
    def __init__(self,fan_in:int,fan_out:int,initial_policy:str = 'random') -> None:
        if initial_policy == 'random':#正态分布初始化
            self.weight = MyTensor(np.random.randn(fan_in,fan_out),requires_grad=True)
            self.bias = MyTensor(np.random.randn(fan_out),requires_grad=True)
        if initial_policy == 'zeros':
            self.weight = MyTensor(np.zeros((fan_in,fan_out)),requires_grad=True)
            self.bias = MyTensor(np.zeros(fan_out),requires_grad=True)
        self.parameters = [self.weight,self.bias]
    def forward(self,x:MyTensor)->MyTensor:
        #检查形状
        if x.data.ndim == 1:
            x.data = x.data.reshape(1,-1)
        myAssert(x.shape[1] == self.weight.shape[0],'shape not match')
        
        matmul=MatMul()
        matmul.forward(x,self.weight)
        add=Sum()
        add.forward(matmul.output,self.bias)
        return add.output
class MLP():
    def __init__(self,input_size:int,hidden_size:int,output_size:int,initial_policy:str = 'random') -> None:
        self.layer1 = MyLinearLayer(input_size,hidden_size,initial_policy)
        self.layer2 = MyLinearLayer(hidden_size,output_size,initial_policy)
        self.parameters = [self.layer1.weight,self.layer1.bias,self.layer2.weight,self.layer2.bias]
    def forward(self,x:MyTensor)->MyTensor:
        return self.layer2.forward(self.layer1.forward(x))
    def __repr__(self) -> str:
        layer1 = 'Layer1:\n' + 'weight:\n' + str(self.layer1.weight) + '\nbias:\n' + str(self.layer1.bias) + '\n'
        layer2 = 'Layer2:\n' + 'weight:\n' + str(self.layer2.weight) + '\nbias:\n' + str(self.layer2.bias) + '\n'
        return 'MLP:'+'\n'+layer1+layer2
if __name__ == "__main__":
    # #测试MyLinearLayer
    # layer = MyLinearLayer(1,3,initial_policy='zeros')
    # x = MyTensor([9],requires_grad=False)
    # y = layer.forward(x)
    # y.backward()
    # print(layer.weight.grad)
    # print(layer.bias.grad)
    #测试MLP
    mlp = MLP(2,3,2,initial_policy='random')
    x = MyTensor(np.array([1,2]),requires_grad=False)
    y = mlp.forward(x)
    y.backward()
    print(mlp)
    
        
        
        