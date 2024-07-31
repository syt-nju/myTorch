#基于numpy 构建基本的Tensor类
from typing import Union, Tuple
import numpy as np
from autograd import no_grad, is_grad_enabled  
#构建基础版Tensor类
#meta 类，定义一些需要的基本属性和简单初始化 
class TensorMeta(type):
    '''
    Tensor 的 meta 类,定义 Tensor 在类最最初始时的初始化
    '''
    def __init__(cls, name, bases, attrs):
        '''
        初始化Tensor类,执行顺序高于子类init
        '''
        super(TensorMeta, cls).__init__(name, bases, attrs)
        cls.data = None #np矩阵
        cls.grad = None 
        cls.requires_grad = False   #作为input的时候不需要梯度，作为其它的时候需要梯度
        cls.shape = None
        cls.device = "cpu" #规定在cpu或者gpu上运行
        cls.dtype = float
        cls.dim = None

# TODO: 实现计算图类
class ComputationalGraph:
    '''
    计算图类，可以理解为一个链表，每个节点是一个操作
    需要实现的方法:
    1. 添加图节点
    2. 释放节点
    '''
    NotImplementedError


class MyTensor(metaclass=TensorMeta):
    '''
    自定义的Tensor类
    '''
    def __init__(self, data, requires_grad=False, device="cpu", dtype=float):
        '''
        初始化Tensor
        '''
        self.data = np.array(data, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.dim = data.ndim
        self.shape = data.shape

        self.requires_grad = requires_grad and is_grad_enabled()
        if self.requires_grad and self.data.dtype != np.float:
            raise TypeError("only float tensor can require gradients")
        if self.requires_grad:
            self.grad = np.zeros_like(data)
        else:
            self.grad = None

    def __repr__(self):
        '''
        重写print方法
        '''
        return f"MyTensor({self.data})"

    def __str__(self):
        '''
        重写print方法
        '''
        return f"MyTensor({self.data})"
    #注意，为了简单处理，我强制要求+-* @所有的操作数都是MyTensor类型，除法被除数必须Tensor类型
    def __add__(self, other):
        '''
        重写加法运算
        '''
        if isinstance(other, MyTensor):
            self.data = self.data + other.data
        else:
            raise TypeError("MyTensor can only add MyTensor")

    def __radd__(self, other):
        '''
        重写反向加法运算
        '''
        return self.__add__(other)

    def __sub__(self, other):
        '''
        重写减法运算
        '''
        if isinstance(other, MyTensor):
            self.data = self.data - other.data
        else:
            raise TypeError("MyTensor can only subtract MyTensor")

    def __rsub__(self, other):
        '''
        重写反向减法运算
        '''
        return self.__sub__(other)

    def __mul__(self, other):
        '''
        重写乘法运算
        这是对应位置相乘，参考numpy的乘法
        '''
        if isinstance(other, MyTensor):
            self.data = self.data * other.data
        else:
            raise TypeError("MyTensor can only multiply MyTensor")

    def __rmul__(self, other):
        '''
        重写反向乘法运算
        '''
        return self.__mul__(other)

    def __matmul__(self, other):
        '''
        重写矩阵乘法运算
        '''
        if isinstance(other, MyTensor):
            self.data = np.matmul(self.data, other.data)
        else:
            raise TypeError("MyTensor can only matmul MyTensor")
    def __truediv__(self, other):
        '''
        重写除法运算
        '''
        if isinstance(other, MyTensor):
            self.data = self.data / other.data
        elif isinstance(other, (int, float)):
            self.data = self.data / other
        else:
            raise TypeError("MyTensor can only divide MyTensor or number")
    def __rtruediv__(self, other):
        '''
        重写反向除法运算
        '''
        return self.__truediv__(other)
    @property
    def transpose(self):
        '''
        转置
        '''
        self.data = np.transpose(self.data)

    @property
    def shape(self):
        '''
        返回 MyTensor 的形状
        '''
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        '''
        返回 MyTensor 的维度
        '''
        return self.data.ndim
    
    @property
    def dtype(self):
        '''
        返回 MyTensor 的数据类型
        '''
        return self.data.dtype



    def __len__(self):
        '''
        返回长度
        '''
        return len(self.data)
    
    def __getitem__(self, key):
        '''
        重写索引
        '''
        if isinstance(key, tuple):
            new_key = []
            for k in key:
                if isinstance(k, MyTensor):
                    new_key.append(k.data)
                else:
                    new_key.append(k)
            key = tuple(new_key)
        elif isinstance(key, MyTensor):
            key = key.data
        return self.data[key]
    

    #TODO: 实现梯度相关的方法
    @property
    def is_leaf(self)->bool:
        '''
        判断是否是叶子节点
        '''
        NotImplementedError

    def backward(self, retain_graph=False):
        '''
        反向传播
        retain_graph: 是否保留计算图
        '''
        NotImplementedError
    
    def zero_grad(self):
        '''
        梯度清零
        '''
        NotImplementedError
    
    def max(self, axis = None, keepdims: bool = False):
        '''
        最大值
        '''
        return np.max(self.data, axis=axis, keepdims=keepdims)

    def sum(self, axis = None, keepdims: bool = False):
        return np.sum(self, axis, keepdims)


            
#TODO: 实现基本的操作类,支持梯度传播
class BinaryOp(MyTensor):
    def __init__(self, x:  MyTensor, y:MyTensor) -> None:
        assert x.device == y.device, "device not match"
        self.device = x.device
        NotImplementedError
    
    def forward(self, x: MyTensor, y: MyTensor)->np.ndarray:
        NotImplementedError

    def backward(self, x: MyTensor, grad: np.ndarray)->np.ndarray:
        '''
        x: 下游节点
        grad: 上游传入该节点的梯度
        '''
        NotImplementedError

class Add(BinaryOp):
    '''
    加法算子, 继承自BinaryOp
    '''

    def forward(self, x: MyTensor, y: MyTensor)->np.ndarray:
        return x.data + y.data

    def backward(self, x: MyTensor, grad: np.ndarray)->np.ndarray:
        return grad[...]
    
# TODO 实现其它常用算子