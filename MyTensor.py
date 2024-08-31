#基于numpy 构建基本的Tensor类
from typing import Union, Tuple,Optional
import numpy as np
from autograd import no_grad, is_grad_enabled 
from utils import myAssert
import random

READSIGN=114514#作为标记，无实际意义
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
    @classmethod
    def index(cls, node):
        '''
        查找节点,返回index
        '''
        return cls.node_list.index(node)

    



class MyTensor(metaclass=TensorMeta):
    '''
    自定义的Tensor类
    '''
    def __init__(self, data:np.ndarray, requires_grad=False, device="cpu", dtype=float):
        '''
        初始化Tensor
        '''
        self.data = data.astype(dtype)
        self.device = device
        self.requires_grad = requires_grad  
        self.father_Op:Op=None

        
        if self.requires_grad and self.data.dtype != float:
            raise TypeError("only float tensor can require gradients")
        if self.requires_grad:
            self.grad = np.zeros_like(data,dtype=dtype)
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
        if self.father_Op==None:
            return True
        else:
            return False

    def backward(self):
        '''
        反向传播
        '''
        #找到Tensor它的产生者的位置
        myAssert(self.father_Op==ComputationalGraph.node_list[-1],"我们强制要求有且仅有一个Tensor作为输出，其生成它的op必须是op list中的最后一个，这里违背了这个规则")
        #把自己的梯度设置为1，用于bp
        self.grad=np.ones_like(self.data)
        for op in ComputationalGraph.node_list[::-1]:
            op.op_backward()
        #清空自己的梯度，最终输出一定不需要梯度
        self.grad=None
    
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
        self.last=list()
        self.output=None

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
            5.记录算子的输出output
        '''
        NotImplementedError

    def grad_func(self, grad: np.ndarray) -> Tuple[np.ndarray]:
        '''
        梯度传播
        '''
        NotImplementedError
    def op_backward(self):
        '''
        单个算子的反向传播
        '''
        for node in self.last:
            if node.requires_grad:
                #handle broadcast
                add_grad=self.grad_func(node,self.output.grad)
                if node.grad.shape!=add_grad.shape:
                    #找到被广播的维度
                    broadcast_axis=[-i for i in range(1,add_grad.ndim+1) if add_grad.shape[-i]==1]     
                    add_grad=np.sum(add_grad,axis=tuple(broadcast_axis),keepdims=True)#求均值压缩
                    if node.grad.ndim<add_grad.ndim:
                        add_grad=np.sum(add_grad,axis=tuple(range(add_grad.ndim-node.grad.ndim)))#把延长的维度进行压缩，这里无所谓sum还是mean，理想状态下应该前几个维度大小都为1
                node.grad+=add_grad
                    
                    



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
        #考虑bias的情况，这里允许出现不同shape的情况，但是要求能够广播
        # for arg in args:
        #     myAssert(arg.shape == args[0].shape, f"{arg}shape must be the same as {args[0]}", arg, args[0])
        myAssert(all(arg.device == self.device for arg in args), "device must be the same",self.device)
        
        #算出结果
        #这里我希望它自己会广播，于是我采用笨一点的硬加的方法
        #a+=b这中用法要保证b是被广播的,调用尽量把大的放前面
        result=args[0].data
        for arg in args[1:]:
            result+=arg.data
        
        
        z = MyTensor(result, requires_grad= not all(not arg.requires_grad for arg in args), device=self.device) #暂定为，当且仅当所有输入的requires_grad=false，输出为requires_grad=false
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
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
        
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
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
class MatMul(Op):
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：shape是否相同;device是否相同
        myAssert(args.__len__()==2, "MatMul must have 2 arguments")
        myAssert(args[0].ndim==2 and args[1].ndim==2,"MatMul must have 2-D tensor")
        myAssert(args[0].shape[1]==args[1].shape[0],"MatMul shape error")
        myAssert(all(arg.device == self.device for arg in args), "device must be the same",self.device)
        
        #算出结果
        self.a_one_dimen,self.b_one_dimen=args[0].ndim<2,args[1].ndim<2#记录是否是一维,由于其一维matmul的特殊性需要进行特殊处理
        #记录下是否出现矩阵一维导致的broadcast
        self.is_a_broadcast,self.is_b_broadcast=args[0].ndim<2,args[1].ndim<2
        
        result=np.matmul(args[0].data,args[1].data)
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node:MyTensor,grad: np.ndarray)->np.ndarray: 
        '''
        此处处理比较多，参考文献
        https://welts.xyz/2022/04/26/broadcast/
        @param
            node: MyTensor 对node求导
            grad: np.ndarray 上游传来的梯度
        @return
            返回对应的导数值，为np.ndarray
        '''
        #规范上游梯度的ndim问题
        if self.is_a_broadcast:
            np.expand_dims(grad,axis=   0)
        if self.is_b_broadcast:
            np.expand_dims(grad,axis=-1)
            
            
        if node==self.last[0]:
            return grad@self.last[1].data.swapaxes(-1,-2)
        elif node==self.last[1]:
            return self.last[0].data.swapaxes(-1,-2)@grad
        else:
            raise ValueError("something wrong,check self.last")

if __name__ == "__main__":
    #测试
    #对整个图的构造进行测试
    #y=ax+b 测试
    
    x=MyTensor(np.array([1,2,3]),requires_grad=False)
    a=MyTensor(np.array([1,1,1]),requires_grad=True)
    b=MyTensor(np.array([7,8,9]),requires_grad=True)
    add=Sum()
    mul=Mul()
    temp=mul.forward(a,x)
    y=add.forward(temp,b)
    print(y)
    print(a.grad,b.grad)
    #z=x(ax+b)
    new_mul=Mul()
    z = new_mul.forward(y, x)
    print(z)
    z.backward()
    print(a.grad,b.grad)
    print("测试结束")
    #测试矩阵乘法
    ComputationalGraph.clear()
    x=MyTensor(np.array([[1,2,3],[4,5,6]]),requires_grad=False)
    y=MyTensor(np.array([[1,2],[1,2],[2,1]]),requires_grad=True)
    mul=MatMul()   
    z=mul.forward(x,y)
    print(z)
    z.backward()
    print(y.grad)