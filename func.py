'''实现所需要的算子'''

import numpy as np
import MyTensor
import torch

def sigmoid(x:MyTensor)->MyTensor:
    '''
    实现sigmoid函数
    '''
    x.data = 1/(1 + np.exp(-x.data))

def softmax(x:MyTensor)->MyTensor:
    '''
    实现softmax函数
    '''
    x_sub_max = x.data - np.max(x.data)
    x.data = np.exp(x_sub_max)/np.sum(np.exp(x_sub_max))

def ReLU(x:MyTensor)->MyTensor:
    '''
    实现relu函数
    '''
    x.data = np.maximum(0, x.data)

def tanh(x:MyTensor)->MyTensor:
    '''
    实现tanh函数
    '''
    x.data = np.tanh(x.data)

def leaky_ReLU(x:MyTensor, alpha: float)->MyTensor:
    '''
    实现leaky_relu函数
    '''
    x.data = np.maximum(x.data, alpha*x.data)

def L1Loss(y_pred:MyTensor, y_true:MyTensor, reduction = 'mean'):
    '''
    实现L1损失函数
    '''
    l1 = np.sum(np.abs(y_pred.data - y_true.data))
    if reduction == 'mean':
        return MyTensor.MyTensor(l1/y_pred.data.size)
    elif reduction == 'sum':
        return MyTensor.MyTensor(l1)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

def MSELoss(y_pred:MyTensor, y_true:MyTensor, reduction = 'mean'):
    '''
    实现均方误差损失函数(MSE)
    '''
    square_sum = np.sum(np.square(y_pred.data - y_true.data))
    if reduction == 'mean':
        return MyTensor.MyTensor(square_sum/y_pred.data.size)
    elif reduction == 'sum':
        return MyTensor.MyTensor(square_sum)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")

def NLL_loss(y_pred:MyTensor, y_true:MyTensor, reduction = 'mean'):
    '''
    实现负对数似然损失函数(NLL)
    '''
    nll = -np.sum(y_true.data*np.log(y_pred.data))
    if reduction == 'mean':
        return MyTensor.MyTensor(nll/y_pred.data.size)
    elif reduction == 'sum':
        return MyTensor.MyTensor(nll)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
   
def CrossEntropyLoss(y_pred:MyTensor, y_true:MyTensor, reduction = 'sum'):
    '''
    实现交叉熵损失函数
    '''
    y_pred_sub_max = y_pred.data - np.max(y_pred.data)
    exp_sum = np.sum(np.exp(y_pred_sub_max))
    log_sum_exp = np.log(exp_sum)
    simplified_cross = log_sum_exp - y_pred_sub_max
    cross_entropy = np.sum(y_true.data * simplified_cross)
    if reduction == 'mean':
        return MyTensor.MyTensor(cross_entropy/y_pred.data.size)
    elif reduction == 'sum':
        return MyTensor.MyTensor(cross_entropy)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
    

if __name__ == "__main__":
    a = MyTensor.MyTensor(np.array([1,2,3,4]))
    b = MyTensor.MyTensor(np.array([3,4,1,-4]))
    print(CrossEntropyLoss(a,b,'sum'))
    print(MSELoss(a,b,'mean'))

    a1 = torch.tensor([1,2,3,4], dtype=torch.float32)
    b1 = torch.tensor([3,4,1,-4], dtype=torch.float32)
    print(torch.nn.CrossEntropyLoss()(a1,b1))

    loss1 = torch.nn.MSELoss()
    input = torch.tensor([1,2,3,4], dtype=torch.float32)
    target = torch.tensor([3,4,1,-4], dtype=torch.float32)
    output = loss1(input, target)
    print(output)
    
    

