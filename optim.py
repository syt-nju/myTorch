import MyTensor
import numpy as np
import my_nn
from loss_func import MSELoss
class BaseOptimizer():
    def __init__(self, parameters, lr=0.01):
        '''
        parameters: list, 待优化的Tensor参数
        '''
        self.parameters = parameters
        self.lr = lr
    def step(self):
        raise NotImplementedError
    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)
class BGD(BaseOptimizer):
    '''
        固定学习率的批量梯度下降
    '''
    def __init__(self,parameters,lr=0.01):
        super().__init__(parameters,lr)
    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad
class SGD(BaseOptimizer):
    '''
        固定学习率的随机梯度下降
    '''
    def __init__(self,parameters,lr=0.01):
        super().__init__(parameters,lr)
    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad
class AdaGrad(BaseOptimizer):
    '''
        自适应学习率的梯度下降，在参数更新不均匀的情况下，可以比较好地自适应调整学习率，一般啥参数不用调
        params: _params_t,
        lr: float = ...,
        lr_decay: float = ...(学习率衰减系数,一般对于AdaGrad，这个参数没什么用，默认值就行),
        weight_decay: float = ...(L2正则化项系数),
        initial_accumulator_value: float = ...,
        eps: float = ...
    '''
    def __init__(self,parameters,lr=0.01,eps=1e-8,lr_decay=0.0,weight_decay=0.0,initial_accumulator_value=0.0):
        super().__init__(parameters,lr)
        self.eps = eps
        self.G = [np.full(param.data.shape,initial_accumulator_value) for param in parameters]
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.lr = lr
    def step(self):
        self.G = [self.G[i]+param.grad**2 for i,param in enumerate(self.parameters)]
        for i  in range(len(self.parameters)):
            param = self.parameters[i]
            param.data -= self.lr/(np.sqrt(self.G[i])+self.eps)*param.grad
        
        self.lr *= 1/(1+self.lr_decay)
        
        
if __name__=='__main__':
    #测试BGD
    # #构造y=3x+2的模拟
    x=np.array(range(10)).reshape(-1,1)
    y_true=3*x+2+np.random.randn(10,1)*0.001
    layer=my_nn.MyLinearLayer(1,1,initial_policy='zeros')
    optimizer=AdaGrad([layer.weight,layer.bias],lr=10)
    loss_func=MSELoss()
    for i in range(1000000):
        y_pred=layer.forward(MyTensor.MyTensor(x))
        loss=loss_func.forward(y_pred,MyTensor.MyTensor(y_true))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if loss.data<1e-5:
            print("epcho",i)
            break
    print(layer.weight.data,layer.bias.data)
    