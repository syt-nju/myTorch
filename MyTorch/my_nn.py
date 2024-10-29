from typing import Tuple
from utils.utils import *
from MyTensor import MyTensor,MatMul,Sum,Op,Mul,Max,Exp,Log,Sub,SumUnary,Div
from MyTensor import ComputationalGraph
import numpy as np
class Sequential():
    def __init__(self,*args) -> None:
        self.layers = list(args)

    def forward(self,x:MyTensor)->MyTensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def __repr__(self) -> str:
        
        res = 'Model structure:\n   Sequential(\n'
        for layer in self.layers:
            res += '        '+str(layer) + '\n'
        res += ')'
        return res
    def __len__(self):
        return len(self.layers)
    def __getitem__(self,index):
        return self.layers[index]
class MyLinearLayer():
    def __init__(self,fan_in:int,fan_out:int,initial_policy:str = 'random') -> None:
        '''
           @fan_in:输入维度
           @fan_out:输出维度
           @initial_policy:初始化策略，可以是'random','zeros,'xavier','He'
        '''
        if initial_policy == 'random':#正态分布初始化
            self.weight = MyTensor(np.random.randn(fan_in,fan_out)*np.sqrt(1/fan_in),requires_grad=True)
            self.bias = MyTensor(np.random.randn(fan_out)*np.sqrt(1/fan_in),requires_grad=True)
        if initial_policy == 'zeros':
            self.weight = MyTensor(np.zeros((fan_in,fan_out)),requires_grad=True)
            self.bias = MyTensor(np.zeros(fan_out),requires_grad=True)
        if initial_policy=='xavier':
            self.weight = MyTensor(np.random.randn(fan_in,fan_out)*np.sqrt(2/fan_in+fan_out),requires_grad=True)
            self.bias = MyTensor(np.random.randn(fan_out)*np.sqrt(2/fan_in+fan_out),requires_grad=True)
        if initial_policy=='He':
            self.weight = MyTensor(np.random.randn(fan_in,fan_out)*np.sqrt(2/fan_in),requires_grad=True)
            self.bias = MyTensor(np.random.randn(fan_out)*np.sqrt(2/fan_in),requires_grad=True)
        self.parameters = [self.weight,self.bias]
    def __str__(self) -> str:
        return 'MyLinearLayer('+str(self.weight.shape[0])+','+str(self.weight.shape[1])+')'
    def __repr__(self):
        return f'weight:\n{self.weight}\nbias:\n{self.bias}'
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
        '''
           initial_policy:初始化策略，可以是'random','zeros,'xavier','He'
        '''
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size        
        self.layer1 = MyLinearLayer(input_size,hidden_size,initial_policy)
        self.layer2 = MyLinearLayer(hidden_size,output_size,initial_policy)
        self.parameters = [self.layer1.weight,self.layer1.bias,self.layer2.weight,self.layer2.bias]
    def forward(self,x:MyTensor)->MyTensor:
        return self.layer2.forward(self.layer1.forward(x))
    def __repr__(self) -> str:
        layer1 = 'Layer1:\n' + 'weight:\n' + str(self.layer1.weight) + '\nbias:\n' + str(self.layer1.bias) + '\n'
        layer2 = 'Layer2:\n' + 'weight:\n' + str(self.layer2.weight) + '\nbias:\n' + str(self.layer2.bias) + '\n'
        return 'MLP:'+'\n'+layer1+layer2
    def __str__(self):
        return f"MLP({self.input_size},{self.hidden_size},{self.output_size})"
class Softmax():#采用小算子的forward来实现计算图的构建
    def __init__(self,dim=0) -> None:
        '''@dim:指定沿哪个维度应用softmax'''
        self.dim = dim
    def forward(self,x:MyTensor)->MyTensor:
        '''     x_sub_max = x.data - np.max(x.data, axis = dim, keepdims = True)
                exp = np.exp(x_sub_max)
                exp_sum = np.sum(exp, axis = dim, keepdims = True)
                x.data = exp/exp_sum'''
        sub_1=Sub()
        max=Max(axis=self.dim,keepdims=True)
        exp_1=Exp()
        sumunary=SumUnary(axis=self.dim,keepdims=True)
        div=Div()
        
        
        x_sub_max = sub_1.forward(x,max.forward(x))
        exp=exp_1.forward(x_sub_max)
        exp_sum=sumunary.forward(exp)
        result=div.forward(exp,exp_sum)
        return result
class ReLU(Op):
    def forward(self,*args)->MyTensor:
        '''    x.data = np.maximum(x.data, 0)'''
        myAssert(args.__len__()==1, "Relu must have 1 arguments")
        
        result = np.maximum(args[0].data, 0)
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node,grad: np.ndarray) -> np.ndarray:
        return grad * (node.data > 0)
    def __repr__(self) -> str:
        return 'ReLU()'
        
    
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
    
        
        
        