import numpy as np
from MyTorch.utils.utils import myAssert
#重构一版,修改计算图进入逻辑以更贴合pytorch
#这版计算图只存tensor
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

class MyTensor():
    '''
    自定义的Tensor类
    
    parameters:
        data: np.ndarray 数据
        requires_grad: bool 是否需要梯度
        device: str 设备
        dtype
    '''
    def __init__(self, data:np.ndarray, requires_grad=False, device="cpu", dtype=float,fatherop=None):
        '''
        初始化Tensor
        '''
        self.data = data.astype(dtype)
        self.device = device
        self.requires_grad = requires_grad  
        self.father_op=fatherop
        self.father_tensor=list()

        
        if self.requires_grad and self.data.dtype != float:
            raise TypeError("only float tensor can require gradients")
        if self.requires_grad:
            self.grad = np.zeros_like(data,dtype=dtype)
            #将自己加入计算图
            ComputationalGraph.add_node(self)
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
        return Sum.forward(self,other)

    def __radd__(self, other):
        '''
        重写反向加法运算
        '''
        return self.__add__(other)

    def __sub__(self, other):
        '''
        重写减法运算
        '''
        return Sub.forward(self,other)

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
        return Mul.forward(self,other)

    def __rmul__(self, other):
        '''
        重写反向乘法运算
        '''
        return self.__mul__(other)

    def __matmul__(self, other):
        '''
        重写矩阵乘法运算
        '''
        return MatMul.forward(self,other)
    def __truediv__(self, other):
        '''
        重写除法运算
        '''
        Div.forward(self,other)
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
        if self.father_op==None:
            return True
        else:
            return False

    def backward(self,retain_graph:bool=False):
        '''
        反向传播
        @param
            retain_graph: bool 是否保留计算图
        '''
        #找到Tensor它的产生者的位置
        myAssert(self==ComputationalGraph.node_list[-1],"此tensor不是计算图最终输出",self)

        self.grad=np.ones_like(self.data).astype(self.data.dtype)
        for tensor in reversed(ComputationalGraph.node_list):
            if not tensor.is_leaf:
                for i in tensor.father_tensor:
                    if i.requires_grad:
                        i.grad+=tensor.father_op.grad_fn(i,tensor.grad,tensor.father_tensor,axis=tensor.axis,keepdims=tensor.keepdims)

        #清空计算图
        if not retain_graph:
            ComputationalGraph.clear()
    
    def zero_grad(self):
        '''
        梯度清零
        '''
        self.grad = np.zeros_like(self.data)
    
    def max(self, axis = None, keepdims: bool = False):
        '''
        最大值
        '''
        return np.max(self.data, axis=axis, keepdims=keepdims)

    def sum(self, axis = None, keepdims: bool = False):
        return np.sum(self, axis, keepdims)
def op_forward(forward_func):
    '''
    修饰器，用于Op的forward方法
    '''
    def wrapper(cls, *args,axis=0,keepdims=False):
        device=args[0].device
        myAssert(all(arg.device == device for arg in args), "device must be the same",device)
        data=forward_func(cls,*args,axis=axis,keepdims=keepdims)
        result = MyTensor(data, requires_grad= not all(not arg.requires_grad for arg in args), device=device)
        result.father_tensor=args
        result.father_op=cls
        result.axis=axis
        result.keepdims=keepdims
        return result
    return wrapper
class Op():
    '''
    Op的元类
    '''
    @classmethod
    def forward(cls, *args):
        '''
        前向传播
        '''
        raise NotImplementedError
    
    @classmethod
    def grad_fn(cls):
        '''
        梯度计算
        '''
        raise NotImplementedError
    

class Sum(Op):
    '''
    加法
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, y:MyTensor,**kwargs):
        '''
        前向传播
        '''
        result_data=x.data+y.data
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        return last_grad
class Sub(Op):
    '''
    减法
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, y:MyTensor,**kwargs):
        '''
        前向传播
        '''
        result_data=x.data-y.data
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        if x==input_tensors[0]:
            return last_grad
        elif x==input_tensors[1]:
            return -last_grad
        else:
            raise ValueError("求导对象不在输入中")
class Mul(Op):
    '''
    乘法
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, y:MyTensor,**kwargs):
        '''
        前向传播
        '''
        result_data=x.data*y.data
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        if x==input_tensors[0]:
            return last_grad*input_tensors[1].data
        elif x==input_tensors[1]:
            return last_grad*input_tensors[0].data
        else:
            raise ValueError("求导对象不在输入中")
class MatMul(Op):
    '''
    矩阵乘法
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, y:MyTensor,**kwargs):
        '''
        前向传播
        '''
        result_data=np.matmul(x.data,y.data)
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        is_a_broadcast,is_b_broadcast=input_tensors[0].ndim<2,input_tensors[1].ndim<2
        if is_a_broadcast:
            last_grad=np.expand_dims(last_grad,axis=0)
        if is_b_broadcast:
            last_grad=np.expand_dims(last_grad,axis=-1)
        
        if x==input_tensors[0]:
            grad=(last_grad@np.atleast_2d(input_tensors[1].data)) if is_b_broadcast else (last_grad@input_tensors[1].data.swapaxes(-1,-2))
            if is_a_broadcast:
                grad=grad[0]
            return grad
        elif x==input_tensors[1]:
            
            grad= np.atleast_2d(input_tensors[0].data).swapaxes(-1,-2)@last_grad if is_a_broadcast else input_tensors[0].data.swapaxes(-1,-2)@last_grad
            if is_b_broadcast:
                grad=[...,0]
            return grad
        else:
            raise ValueError("求导对象不在输入中")
class Div(Op):
    '''
    除法
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, y:MyTensor,**kwargs):
        '''
        前向传播
        '''
        result_data=x.data/y.data
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        if x==input_tensors[0]:
            return last_grad/input_tensors[1].data
        elif x==input_tensors[1]:
            return -last_grad*input_tensors[0].data/input_tensors[1].data**2
        else:
            raise ValueError("求导对象不在输入中")
class Max(Op):
    '''
    最大值
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, **kwargs):
        '''
        前向传播
        params:
            x: MyTensor 输入
            axis: int 求最大值的轴
            keepdims: bool 是否保留维度
        '''
        assert "axis" in kwargs, "kwargs must contain 'axis'"
        assert "keepdims" in kwargs, "kwargs must contain 'keepdims'"
        result_data=np.max(x.data, axis=kwargs["axis"], keepdims=kwargs["keepdims"])
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        axis=kwargs["axis"]
        keepdims=kwargs["keepdims"]
        if not keepdims:
            last_grad=np.expand_dims(last_grad,axis=axis)
        mask=np.equal(x.data,np.max(x.data,axis=axis,keepdims=True))
        return last_grad*mask
class Exp(Op):
    '''
    指数
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, **kwargs):
        '''
        前向传播
        '''
        result_data=np.exp(x.data)
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        return last_grad*np.exp(x.data)
class Log(Op):
    '''
    对数
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, **kwargs):
        '''
        前向传播
        '''
        result_data=np.log(x.data)
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        return last_grad/x.data
class SumUnary(Op):
    '''
    求和
    '''
    @classmethod
    @op_forward
    def forward(cls, x:MyTensor, **kwargs):
        '''
        前向传播
        @param
            x: MyTensor 输入
            axis: int 求和的轴
            keepdims: bool 是否保留维度
        '''
        result_data=np.sum(x.data,axis=kwargs["axis"],keepdims=kwargs["keepdims"])
        return result_data
    @classmethod
    def grad_fn(cls,x:MyTensor,last_grad:np.ndarray,input_tensors:list[MyTensor],**kwargs):
        '''
        梯度计算
        params:
            x: MyTensor 求导对象
            last_grad: np.ndarray 上游梯度
            input_tensors: list[MyTensor] 函数输入的tensor
        '''
        if not (kwargs["axis"] is None or kwargs["keepdims"]):
            last_grad=np.expand_dims(last_grad,axis=kwargs["axis"])
        return last_grad*np.ones_like(x.data)