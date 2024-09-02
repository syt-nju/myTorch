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
            param.grad
            param.data -= self.lr*param.grad
        
if __name__=='__main__':
    #测试BGD
    # #构造y=3x+2的模拟
    x=np.array(range(10)).reshape(-1,1)
    y_true=3*x+2+np.random.randn(10,1)*0.001
    layer=my_nn.MyLinearLayer(1,1,initial_policy='zeros')
    optimizer=BGD([layer.weight,layer.bias],lr=0.01)
    for i in range(1000):
        y_pred=layer.forward(MyTensor.MyTensor(x))
        
        loss=MSELoss().forward(y_pred,MyTensor.MyTensor(y_true))
        loss.backward()
        print("grad",layer.weight.grad,layer.bias.grad)
        optimizer.step()
        optimizer.zero_grad()
    print(layer.weight.data,layer.bias.data)
    