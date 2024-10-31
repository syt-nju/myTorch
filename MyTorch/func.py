'''实现所需要的算子'''

import numpy as np
from MyTorch import MyTensor


def sigmoid(x:MyTensor)->MyTensor:
    '''
    实现sigmoid函数
    '''
    x.data = 1/(1 + np.exp(-x.data))

def softmax(x:MyTensor, dim = 0)->MyTensor:
    '''
    实现softmax函数
    '''
    x_sub_max = x.data - np.max(x.data, axis = dim, keepdims = True)
    exp = np.exp(x_sub_max)
    exp_sum = np.sum(exp, axis = dim, keepdims = True)
    x.data = exp/exp_sum
    #x.data=np.exp(x.data)/np.sum(np.exp(x.data)),采用上述形式防止溢出

def log_softmax(x: MyTensor, axis = None, keepdims = True):
    x_sub_max = x.data - np.max(x.data, axis = axis, keepdims = keepdims)
    x.data =  x_sub_max - np.log(np.sum(np.exp(x_sub_max), axis = axis, keepdims = keepdims))

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

def leaky_ReLU(x:MyTensor, alpha: float=0.001)->MyTensor:
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

def MSELoss(y_pred: MyTensor, y_true: MyTensor, reduction='mean'):
    '''
    实现均方误差损失函数 (MSE)
    '''
    # 将数据转换为更高精度的浮点数类型（float64）
    y_pred_data = y_pred.data.astype(np.float64)
    y_true_data = y_true.data.astype(np.float64)
    
    # 计算平方和
    square_sum = np.sum(np.square(y_pred_data - y_true_data))
    
    # 根据 reduction 参数返回结果
    if reduction == 'mean':
        return MyTensor.MyTensor(square_sum / y_pred_data.size)
    elif reduction == 'sum':
        return MyTensor.MyTensor(square_sum)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")


def NLL_loss(y_pred:MyTensor, y_true:MyTensor, reduction = 'mean'):
    '''
    实现负对数似然损失函数(NLL)
    '''
    if y_pred.ndim == 1:
        nll = - y_pred.data[y_true.data[0].astype(int)]
    elif y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred and y_true must have the same shape")
    else:
        nll = - y_pred.data[np.arange(y_pred.data.shape[0]), y_true.data.astype(int)]
    if reduction == 'mean':
        return MyTensor.MyTensor(np.mean(nll))
    elif reduction == 'sum':
        return MyTensor.MyTensor(np.sum(nll))
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
    #批量测试
    import torch
    from utils.utils import myAssert
    from utils.utils import random_perturbation
    #构造函数list
    single_input_funcs=[sigmoid,softmax,ReLU,tanh,leaky_ReLU]#单输入函数.leaky_ReLU函数需要额外参数alpha,默认为1e-3
    double_input_funcs=[L1Loss,MSELoss,NLL_loss,CrossEntropyLoss]#双输入函数
    
    #构造数据
    np.random.seed(0)
    epcho=100
    #单输入函数的测试
    for i in range(epcho):
        X=np.random.randn(3,3)
        for func in single_input_funcs:
            result=MyTensor.MyTensor(X)
            func(result)
            result_np=result.data
            if func==sigmoid:
                result_torch=torch.sigmoid(torch.tensor(X))
            elif func==softmax:
                result_torch=torch.softmax(torch.tensor(X),dim=0)
            elif func==ReLU:
                result_torch=torch.relu(torch.tensor(X))
            elif func==tanh:
                result_torch=torch.tanh(torch.tensor(X))
            elif func==leaky_ReLU:
                result_torch=torch.nn.functional.leaky_relu(torch.tensor(X),negative_slope=1e-3)
            else:
                raise ValueError("未知的函数")
            judge=np.allclose(result_np,result_torch.numpy(),atol=1e-5)
            myAssert(judge,f"{func.__name__}函数,在第{i}轮测试失败",result_np,result_torch.numpy())
    print("单输入检查完成")
    #双输入检查还没写        
    for i in range(epcho):
        #随机生成0，1，2 三种类型标签
        y_true=np.random.randint(0,3,(3,))
        #对true进行扰动构造预测标签
        y_pred=random_perturbation(y_true,(0,3),0.2)#以0.2的概率对每一项进行扰动
        
        for func in double_input_funcs:
            result=func(MyTensor.MyTensor(y_pred),MyTensor.MyTensor(y_true),reduction='sum').data
            if func == L1Loss:
                result_torch = torch.nn.functional.l1_loss(torch.tensor(y_pred), torch.tensor(y_true), reduction='sum')
            elif func == MSELoss:
                result_torch = torch.nn.functional.mse_loss(torch.tensor(y_pred.astype(float)), torch.tensor(y_true.astype(float)), reduction='sum')
            elif func == NLL_loss:
                result_torch = torch.nn.functional.nll_loss(torch.tensor(y_pred.astype(float)), torch.tensor(y_true).long(), reduction='sum')
            elif func == CrossEntropyLoss:
                result_torch = torch.nn.functional.cross_entropy(torch.tensor(y_pred.astype(float)), torch.tensor(y_true.astype(float)), reduction='sum')
            else:
                raise ValueError("Unknown function")
            judge = np.allclose(result, result_torch.numpy(), atol=1e-5)
            myAssert(judge, f"{func.__name__} function failed in round {i}", result, result_torch.numpy(),y_true,y_pred)
    
# X = MyTensor.MyTensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
# y = MyTensor.MyTensor(np.array([0,1,2]))
# print(X.shape[0] == y.shape[0])

# print(NLL_loss(X, y, reduction = 'mean'))

# X1 = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32)
# y1 = torch.tensor([0,1], dtype = torch.long)
# print(torch.nn.functional.nll_loss(X1, y1, reduction = 'mean'))