from MyTorch.MyTensor import MyTensor
import numpy as np
from MyTorch import my_nn
from MyTorch.loss_func import MSELoss
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
    def __init__(self,parameters,lr=1e-3):
        super().__init__(parameters,lr)
    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad
class SGD(BaseOptimizer):
    '''
        包含多个参数的随机梯度下降
        @params: _params_t,
        lr: float = ..., 初始学习率
        momentum: float = ..., 动量系数
        weight_decay: float = ..., L2正则化项系数
        nesterov: bool = ..., 是否使用Nesterov动量
    '''
    def __init__(self,parameters,lr=1e-3,momentum=0.0,weight_decay=0.0,nesterov=False):
        super().__init__(parameters,lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.v = [np.zeros_like(param.data) for param in parameters] #动量
    def step(self):
        for i,param in enumerate(self.parameters):
            #pytorch原汁原味写法，和动量法论文表述不太一样，有点恶心心
            grad = param.grad+param.data*self.weight_decay
            self.v[i] = self.momentum*self.v[i]+grad
            if self.nesterov:
                grad+=self.momentum*self.v[i]
            param.data -= grad*self.lr
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
    def __init__(self,parameters,lr=1e-3,eps=1e-8,lr_decay=0.0,weight_decay=0.0,initial_accumulator_value=0.0):
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
class Adam(BaseOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        '''
            params: _params_t,
            lr: float = ...,初始学习率
            betas: Tuple[float, float] = ...,beta1和beta2的系数,分别对应梯度的一阶矩和二阶矩估计
            eps: float = ...,防止除0的极小数
            weight_decay: float = ...,L2正则化项系数
        '''      
        self.parameters = params
        self.m=[np.zeros_like(param.data) for param in params]#一阶矩估计
        self.v=[np.zeros_like(param.data) for param in params]#二阶矩估计
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)#默认值参考Adam论文
        for key, value in defaults.items():
            setattr(self, key, value)
    
    def step(self):
        for i,param in enumerate(self.parameters):
            grad=param.grad+param.data*self.weight_decay
            #update 矩估计
            self.m[i] = self.betas[0]*self.m[i]+(1-self.betas[0])*grad
            self.v[i] = self.betas[1]*self.v[i]+(1-self.betas[1])*(grad**2)
            #bias correction
            m_hat = self.m[i]/(1-self.betas[0]**(i+1))
            v_hat = self.v[i]/(1-self.betas[1]**(i+1))
            #update param
            param.data -= self.lr*m_hat/(np.sqrt(v_hat)+self.eps)
        
    
if __name__=='__main__':
    #测试BGD
    # #构造y=3x+2的模拟
    x=np.array(range(10)).reshape(-1,1)
    y_true=3*x+2+np.random.randn(10,1)*0.001
    layer=my_nn.MyLinearLayer(1,1,initial_policy='zeros')
    optimizer=SGD([layer.weight,layer.bias],momentum=0.9,nesterov=True)
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
    
    '''
    统一初始学习率下
        目前收敛轮数：
            BGD:9886
            Adam:18253
            动量系数为0.9的SGD:9915
            动量系数为0.9且Nesterov的SGD:942
            
    '''