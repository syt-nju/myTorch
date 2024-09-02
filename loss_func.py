#实现损失函数，forward基于已经实现的func.py的函数
from typing import Tuple
from MyTensor import MyTensor,Op,ComputationalGraph
import func
import numpy as np
class LossFunc(Op):
    #重载op_backward
    def op_backward(self):
        '''
        单个算子的反向传播
        '''
        node =self.last[0]#y_pred
        #由于loss函数只有一个输入，没有上级传入梯度等参数，不会出现广播
        node.grad+=self.grad_func()
       
class MSELoss(LossFunc):
    def forward(self,y_pred:MyTensor, y_true:MyTensor, reduction = 'mean')->MyTensor:
        '''
        实现均方误差损失函数(MSE)
        @return:MyTensor 结果的MyTensor形式
        '''
        if isinstance(y_true,np.ndarray):
            y_true=MyTensor(y_true,requires_grad=False)
        #检查
        assert y_pred.shape == y_true.shape,'shape not match'
        assert y_pred.requires_grad==True,'y_pred must requires_grad'
        assert y_true.requires_grad==False,'y_true must not requires_grad'
        
        #记录此参数用于求grad_func时分类讨论
        self.reduction=reduction
        
        result=func.MSELoss(y_pred,y_true,reduction)
        ComputationalGraph.add_node(self)
        result.father_Op = self
        self.last=[y_pred,y_true]#顺序严格要求
        self.output=result
        return result
    def grad_func(self)->np.ndarray:
        '''
        计算梯度，仅返回值，不进行bp
        '''
        if self.reduction == 'mean':
            return 2*(self.last[0].data-self.last[1].data)/self.last[0].data.size
        elif self.reduction == 'sum':
            return 2*(self.last[0].data-self.last[1].data)
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
        
if __name__ == "__main__":
    # #简单测试
    # y_pred = MyTensor(np.array([1,2,3,3]),requires_grad=True)
    # y_true = np.array([1,2,3,4])
    # loss=MSELoss()
    # result=loss.forward(y_pred,y_true)
    # result.backward()
    # print(y_pred.grad)
    
    #基于pytorch对比的测试
    import torch
    import torch.nn as nn
    import my_nn
    x=np.array(range(10)).reshape(-1,1)
    y_true=3*x+2#+np.random.randn(10,1)*0.001
    layer=my_nn.MyLinearLayer(1,1,initial_policy='zeros')
    y_pred=layer.forward(MyTensor(x,requires_grad=True))
    loss=MSELoss().forward(y_pred,MyTensor(y_true,requires_grad=False))
    loss.backward()
    print("grad",layer.weight.grad,layer.bias.grad)
    

        