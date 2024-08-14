#基于numpy 构建基本的Tensor类
from typing import Union, Tuple,Optional
import numpy as np
from autograd import no_grad, is_grad_enabled 
from utils import myAssert
READSIGN=114514
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
        

#目前就是个list，以后可能会加入更多的属性
class ComputationalGraph():
    node_list=list()
    def __init__(self):
        '''
        初始化计算图
        '''
        super().__init__()
    @classmethod
    def add_node(cls, node):
        '''
        添加节点
        '''
        cls.node_list.append(node)
    @classmethod
    def clear(cls):
        '''
        清空计算图
        '''
        cls.node_list.clear()

    



class MyTensor(metaclass=TensorMeta):
    '''
    自定义的Tensor类
    '''
    def __init__(self, data:np.array, requires_grad=False, device="cpu", dtype=float):
        '''
        初始化Tensor
        '''
        self.data = np.array(data, dtype=dtype)
        self.device = device
        self.requires_grad = requires_grad  
        self.father_Op:Op=None
        
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
    #注意，此时的计算是直接对data进行操作，不涉及梯度传播
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
        return self

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
        判断是否是叶子节点(基于前向传播的DAG图)
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
#区别于学长的torch，我通读代码后，发现实际上算子并不需要继承自Tensor，实际使用的时候，算子只需要对device，require_grad进行要求即可
#所以这里新建一个算子类，无需继承自算子类
class Op:
    '''
    算子类
    '''
    device='cpu'
    def __init__(self, device: str = "cpu", requires_grad: bool = False) -> None:
        self.requires_grad = requires_grad
        self.last=list()

    @classmethod
    def change_device(cls, device: str) -> None:
        '''
        改变算子的设备
        '''
        cls.device = device
    def forward(self, *args: Tuple[MyTensor]) -> MyTensor:
        '''
        forward 流程：
            1.计算并创建最新的Tensor
            2.将当前算子添加到计算图中
            3.添加最新的Tensor的father_Op属性
            4.记录算子的输入到self.last
        '''
        NotImplementedError

    def grad_func(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        '''
        梯度传播
        '''
        NotImplementedError


class Sum(Op):
    '''
    加法算子, 继承自BinaryOp
    '''
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：shape是否相同;device是否相同
        myAssert(args.__len__() > 1, "Sum must have at least 2 arguments")
        for arg in args:
            myAssert(arg.shape == args[0].shape, f"{arg}shape must be the same as {args[0]}", arg, args[0])
        myAssert(all(arg.device == self.device for arg in args), "device must be the same",self.device)
        
        #算出结果
        result=np.zeros_like(args[0].data)
        for arg in args:
            result+=arg.data
        z = MyTensor(result, requires_grad=self.requires_grad, device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        return z
    def grad_func(self, node:MyTensor,grad: np.ndarray)->np.ndarray: 
        '''
        @param
            node: MyTensor 对node求导
            grad: np.ndarray 上游传来的梯度
        @return
            返回对应的导数值，为np.ndarray
        '''
        return grad*np.ones_like(node.data)
class Mul(Op):
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：shape是否相同;device是否相同
        myAssert(args.__len__()==2, "Mul must have 2 arguments")
        for arg in args:
            myAssert(arg.shape == args[0].shape, f"{arg}shape must be the same as {args[0]}", arg, args[0])
        myAssert(all(arg.device == self.device for arg in args), "device must be the same",self.device)
        
        #算出结果
        result=args[0].data*args[1].data
        z = MyTensor(result, requires_grad=self.requires_grad, device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        return z
    def grad_func(self, node:MyTensor,grad: np.ndarray)->np.ndarray: 
        '''
        @param
            node: MyTensor 对node求导
            grad: np.ndarray 上游传来的梯度
        @return
            返回对应的导数值，为np.ndarray
        '''
        if node==self.last[0]:
            return grad*self.last[1].data
        elif node==self.last[1]:
            return grad*self.last[0].data
        else:
            raise ValueError("something wrong,check self.last")
if __name__ == "__main__":
    #测试
    # Test MyTensor class
    data = np.array([1, 2, 3])
    tensor = MyTensor(data)
    print(tensor)  # Output: MyTensor([1 2 3])

    # Test addition operation
    a = MyTensor(np.array([1, 2, 3]))
    b = MyTensor(np.array([4, 5, 6]))
    c= MyTensor(np.array([7, 8, 9]))
    add_op = Sum()
    result = add_op.forward(a,b,c)
    print(result)  # Output: MyTensor([5 7 9])
    print(result.requires_grad)  # Output: False
    print(result.device)  # Output: cpu
    print(result.father_Op)  # Output: None
    print(add_op.last)  # Output: [a, b, c]
    print(ComputationalGraph.node_list)  # Output: [add_op]
