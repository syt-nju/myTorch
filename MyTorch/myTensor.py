#基于numpy 构建基本的Tensor类
from typing import Union, Tuple,Optional
import numpy as np
from MyTorch.autograd import no_grad, is_grad_enabled 
from MyTorch.utils import *
import random
import warnings

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

    def backward(self,retain_graph:bool=False):
        '''
        反向传播
        @param
            retain_graph: bool 是否保留计算图
        '''
        #找到Tensor它的产生者的位置
        myAssert(self.father_Op==ComputationalGraph.node_list[-1],"我们强制要求有且仅有一个Tensor作为输出，其生成它的op必须是op list中的最后一个，这里违背了这个规则")
        #把自己的梯度设置为1，用于bp
        self.grad=np.ones_like(self.data)
        for op in ComputationalGraph.node_list[::-1]:
            op.op_backward()
        #清空自己的梯度，最终输出一定不需要梯度
        self.grad=None
        
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

    def grad_func(self, grad: np.ndarray) -> np.ndarray:
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
                    broadcast_axis=[-i for i in range(1,node.grad.ndim+1) if node.grad.shape[-i]==1]     
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
class Sub(Op):
    def forward(self, *args) -> MyTensor:
        #检查：只能有两个参数;shape是否相同;device是否相同
        myAssert(args.__len__()==2, "Sub must have exact 2 arguments")
        # myAssert(args[0].shape == args[1].shape, f"{args[0]}shape must be the same as {args[1]}", args[0], args[1])
        myAssert(all(arg.device == self.device for arg in args), "device must be the same",self.device)
        
        #算出结果
        result=args[0].data-args[1].data
        
        z = MyTensor(result, requires_grad= not all(not arg.requires_grad for arg in args), device=self.device) #暂定为，当且仅当所有输入的requires_grad=false，输出为requires_grad=false
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node:MyTensor,grad: np.ndarray) ->np.ndarray: 
        if node==self.last[0]:
            return grad
        elif node==self.last[1]:
            return -grad
        else:
            raise ValueError("something wrong,check self.last")
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
class Div(Op):
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：只能有两个参数;shape是否相同;device是否相同
        myAssert(args.__len__()==2, "Div must have exact 2 arguments")
        myAssert(((args[0].shape == args[1].shape)or (len(args[0].data.shape)==len(args[1].data.shape))), f"{args[0]}shape/dim must be the same as {args[1]}", args[0], args[1])
        myAssert(all(arg.device == self.device for arg in args), "device must be the same",self.device)
        
        #算出结果
        result=args[0].data/args[1].data
        
        z = MyTensor(result, requires_grad= not all(not arg.requires_grad for arg in args), device=self.device) #暂定为，当且仅当所有输入的requires_grad=false，输出为requires_grad=false
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node:MyTensor,grad: np.ndarray) ->np.ndarray: 
        if node==self.last[0]:
            return grad/self.last[1].data
        elif node==self.last[1]:
            return -grad*self.last[0].data/(self.last[1].data**2)
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
        
        #记录下是否出现矩阵一维导致的broadcast
        self.is_a_broadcast,self.is_b_broadcast=args[0].ndim<2,args[1].ndim<2
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # 捕捉所有警告,主要目的为捕捉溢出
            result=np.matmul(args[0].data,args[1].data)
            if w:
                for warning in w:
                    print(f"Warning: {warning.message}")
                    print("arg[0]:",args[0].data)
                    print("arg[1]:",args[1].data)

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
            np.expand_dims(grad,axis=0)
        if self.is_b_broadcast:
            np.expand_dims(grad,axis=-1)
            
            
        if node==self.last[0]:
            return grad@self.last[1].data.swapaxes(-1,-2)
        elif node==self.last[1]:
            return self.last[0].data.swapaxes(-1,-2)@grad
        else:
            raise ValueError("something wrong,check self.last")
class Max(Op):
    '''
        max算子
        实例化时可以指定axis和keepdims
    '''
    def __init__(self, device: str = "cpu", requires_grad: bool = False,axis=None,keepdims=False) -> None:
        super().__init__(device, requires_grad)
        self.axis=axis
        self.keepdims=keepdims
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：检查参数是否唯一
        myAssert(args.__len__()==1, "Max must have 1 arguments")
        
        #算出结果
        result=np.max(args[0].data,axis=self.axis,keepdims=self.keepdims)
        
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node,grad: np.ndarray) -> np.ndarray:
        '''参数node会被忽略，因为max是一个单输入的op'''
        #由于所有的比较都要求比较的位置对应，为增强鲁棒性，直接把output的data扩展到和input一样的dim
        if self.keepdims:
            y_full_dim=self.output.data
        else:
            y_full_dim=np.expand_dims(self.output.data,axis=self.axis)
        return (np.isclose(self.last[0].data,y_full_dim,atol=1e-8)).astype(float)*grad
class Exp(Op):
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：检查参数是否唯一
        myAssert(args.__len__()==1, "exp must have 1 arguments")
        
        #算出结果
        result=np.exp(args[0].data)
        
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node,grad: np.ndarray) -> np.ndarray:
        '''参数node会被忽略，因为exp是一个单输入的op'''
        return np.exp(self.last[0].data)*grad
class Log(Op):
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：检查参数是否唯一
        myAssert(args.__len__()==1, "log must have 1 arguments")
        
        #算出结果
        result=np.log(args[0].data)
        
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node,grad: np.ndarray) -> np.ndarray:
        '''参数node会被忽略，因为log是一个单输入的op'''
        return grad/self.last[0].data
class SumUnary(Op):
    def __init__(self, device = "cpu", requires_grad = False,axis=None,keepdims=False) -> None:
        super().__init__(device, requires_grad)
        self.axis=axis
        self.keepdims=keepdims
    def forward(self, *args) -> MyTensor:
        '''
        前向传播
        '''
        #检查：检查参数是否唯一
        myAssert(args.__len__()==1, "sum must have 1 arguments")
        
        #算出结果
        result=np.sum(args[0].data,axis=self.axis,keepdims=self.keepdims)
        
        z = MyTensor(result,requires_grad= not all(not arg.requires_grad for arg in args), device=self.device)
        ComputationalGraph.add_node(self)
        z.father_Op = self
        self.last.extend(list(args))
        self.output=z
        return z
    def grad_func(self, node,grad: np.ndarray) -> np.ndarray:
        '''参数node会被忽略，因为sum是一个单输入的op'''
        if not (self.axis is None or self.keepdims):
            grad=np.expand_dims(grad,axis=self.axis)
        return np.ones_like(self.last[0].data)*grad
if __name__ == "__main__":
    #利用torch,对+，-，*，@进行测试
    import torch
    a = MyTensor(np.random.randn(1, 3), requires_grad=True)
    b = MyTensor(np.random.randn(1, 3), requires_grad=True)
    c = MyTensor(np.random.randn(1, 3), requires_grad=True)
    d = MyTensor(np.random.randn(1, 3), requires_grad=True)
    e = MyTensor(np.random.randn(1, 3), requires_grad=True)
    ops = ['mul', 'sum', 'sum','sub']
    def torch_result(*inputs, ops):
        input_torch = [torch.tensor(i, dtype=torch.float32, requires_grad=True) for i in inputs]
        result_torch = input_torch[0]

        for i, op in enumerate(ops):
            if op == 'mul':
                result_torch = result_torch * input_torch[i + 1]
            elif op == 'sum':
                result_torch = result_torch + input_torch[i + 1]
            elif op == 'sub':
                result_torch = result_torch - input_torch[i + 1]
            else:
                raise ValueError(f"Unsupported operation: {op}")
        result_torch.sum().backward()
        return result_torch, [i.grad.numpy() for i in input_torch]

    def test(ops, *my_tensors):
        result = my_tensors[0]
        for i, op in enumerate(ops):
            if op == 'mul':
                result = Mul().forward(result, my_tensors[i + 1])
            elif op == 'sum':
                result = Sum().forward(result, my_tensors[i + 1])
            elif op == 'sub':
                result = Sub().forward(result, my_tensors[i + 1])
            else:
                raise ValueError(f"Unsupported operation: {op}")

        result.backward()
        my_tensor_data = [tensor.data for tensor in my_tensors]
        torch_result_val, torch_grads = torch_result(*my_tensor_data, ops=ops)

        assert np.allclose(result.data, torch_result_val.detach().numpy(), atol=1e-5), f"Results do not match! Custom = {result.data}, Torch = {torch_result_val.detach().numpy()}"
        for i, my_tensor in enumerate(my_tensors):
            assert np.allclose(my_tensor.grad, torch_grads[i], atol=1e-5), f"Gradients do not match for input {i}! Custom = {my_tensor.grad}, Torch = {torch_grads[i]}"

    test(ops, a, b, c, d,e)
    print("所有结果和梯度匹配！")


    def torch_resultmat(*input, op):
        input_torch = [torch.tensor(i, dtype=torch.float32, requires_grad=True) for i in input]
        result_torch = op(*input_torch)
        result_torch.sum().backward()
        return [i.grad.numpy() for i in input_torch]
    for _ in range(100):
        A = MyTensor(np.random.randn(3,3), requires_grad=True)
        B = MyTensor(np.random.randn(3,3), requires_grad=True)
        matmul = MatMul()
        result = matmul.forward(A, B)
        result.backward()
        result_torch = torch_resultmat(A.data, B.data, op=torch.matmul)
        assert np.allclose(A.grad, result_torch[0], atol = 1e-5), f"MatMul gradients do not match! Custom = {A.grad}, Torch = {result_torch[0]}"
        assert np.allclose(B.grad, result_torch[1], atol = 1e-5), f"MatMul gradients do not match! Custom = {B.grad}, Torch = {result_torch[1]}"
    
    print("MatMul 梯度匹配！")
    
    #测试max
    a = np.array([[ 1.27920267, -1.49798259, -0.95754972],
              [-3.24696494,  1.27485179,  0.34984062],
              [-0.21765164 ,-0.13258395 ,-0.16691512]])
    print(a)
    a=MyTensor(a,requires_grad=True)
    max=Max(axis=1,keepdims=False)
    result=max.forward(a)
    print(result)
    result.backward()
    print(a.grad)