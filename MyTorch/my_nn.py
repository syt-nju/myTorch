from MyTorch.myTensor import MyTensor,MyTensor,MatMul,Sum,Mul,Max,Exp,Log,Sub,SumUnary,Div
from MyTorch.myTensor import ComputationalGraph,op_forward,Op
from MyTorch.utils.utils import *
import numpy as np
class ModuleBase():
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
class Sequential(ModuleBase):
    def __init__(self,*args) -> None:
        self.layers = list(args)
        self.parameters = []
        for layer in self.layers:
            if hasattr(layer,'parameters'):
                self.parameters.extend(layer.parameters)
    def forward(self,x:MyTensor)->MyTensor:
        for layer in self.layers:
            x = layer.forward(x)
            # print(f"layer:{layer.__class__.__name__},output:{x}")
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
    def __call__(self,x:MyTensor)->MyTensor:
        return self.forward(x)
    
class MyLinearLayer(ModuleBase):
    def __init__(self,fan_in:int,fan_out:int,initial_policy:str = 'xavier') -> None:
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
        self.bias.data = self.bias.data.reshape(1,-1)
        if self.bias.requires_grad:
            self.bias.grad = self.bias.grad.reshape(1,-1)
        self.parameters = [self.weight,self.bias]
    def __str__(self) -> str:
        return 'MyLinearLayer('+str(self.weight.shape[0])+','+str(self.weight.shape[1])+')'
    def __repr__(self):
        return f'weight:\n{self.weight}\nbias:\n{self.bias}'
    def forward(self,x:MyTensor)->MyTensor:
        #检查形状
        if x.data.ndim == 1:
            x.data = x.data.reshape(1,-1)
            if x.requires_grad:
                x.grad = x.grad.reshape(1,-1)
        myAssert(x.shape[1] == self.weight.shape[0],'shape not match')
        temp=MatMul.forward(x,self.weight)
        result=Sum.forward(temp,self.bias)
        return result
class Softmax(ModuleBase):#采用小算子的forward来实现计算图的构建
    def __init__(self,dim=0) -> None:
        '''@dim:指定沿哪个维度应用softmax'''
        self.dim = dim
    def forward(self,x:MyTensor)->MyTensor:
        '''     x_sub_max = x.data - np.max(x.data, axis = dim, keepdims = True)
                exp = np.exp(x_sub_max)
                exp_sum = np.sum(exp, axis = dim, keepdims = True)
                x.data = exp/exp_sum'''
        
        x_sub_max = Sub.forward(x,Max.forward(x,axis=self.dim,keepdims=True))
        exp=Exp.forward(x_sub_max)
        exp_sum=SumUnary.forward(exp,axis=self.dim,keepdims=True)
        result=Div.forward(exp,exp_sum)
        return result
class LogSoftmax(ModuleBase):
    def __init__(self,dim=0) -> None:
        self.dim = dim
    def forward(self,x:MyTensor)->MyTensor:
        '''        x_sub_max = x.data - np.max(x.data, axis = axis, keepdims = keepdims)
    x.data =  x_sub_max - np.log(np.sum(np.exp(x_sub_max), axis = axis, keepdims = keepdims))'''
        
        
        x_sub_max = Sub.forward(x,Max.forward(x,axis=self.dim,keepdims=True))
        exp=Exp.forward(x_sub_max)
        exp_sum=SumUnary.forward(exp,axis=self.dim,keepdims=True)
        result = Sub.forward(x_sub_max,Log.forward(exp_sum))
        return result
class ReLU(Op):
    @op_forward
    def forward(self,*args,**kwargs)->MyTensor:
        '''    x.data = np.maximum(x.data, 0)'''
        result=np.maximum(args[0],0)
        return result  
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''    grad = np.where(x.data > 0, grad, 0)'''
        result = np.where(x.data>0,last_grad,0)     
        return result
    def __repr__(self) -> str:
        return 'ReLU()'