# debug 日志
## 2024.8.1

### Bug
**func.py 中的 nll 函数出问题**

测试函数运行结果：

```
(pytor-basic) PS C:\Users\nju22\Desktop\myTorch> & C:/conda/envs/pytor-basic/python.exe c:/Users/nju22/Desktop/myTorch/func.py
单输入检查完成
-1.3862943611198906     #你写的nll的输出
-1.0					#torch的nll的输出
[2 1 1]  				
[2 1 1]					#y_true和y_pred的值，详情请见myAssert的调用处
Traceback (most recent call last):
  File "c:\Users\nju22\Desktop\myTorch\func.py", line 148, in <module>
    myAssert(judge, f"{func.__name__} function failed in round {i}", result, result_torch.numpy(),y_true,y_pred)
  File "c:\Users\nju22\Desktop\myTorch\utils.py", line 19, in myAssert
    assert judge, message
AssertionError: NLL_loss function failed in round 0
```

### bugfix: 对 NLL 损失函数的定义理解有误

之前以为负对数的计算需要在 NLL 函数里面完成，但是实际上 NLL 函数的输入已经是对数似然概率了。即:

> Input is expected to be log-probabilities.

我们可以将 NLL 的输出表示为:

$$
\mathcal{l}(x,y) = L = \{l_1,\cdots, l_N\}^T,\, l_n = -x_{n,y_n}
$$

**Example**

```
input:
y_pred: [ [1,2,3]
          [4,5,6]
          [7,8,9] ]
y_true: [0,1,2]

Output: -5.0 (-(1+5+9)/3, reduction = 'mean')
```

并且需要注意的是 input 的格式:

- y_pred(Tensor): $(N,C)$.
- y_true(Tensor): $(N)$ where each value is in range $[0, C-1]$.

倘若 `y_pred.shape[0]` 与 `y_true.shape[0]` 不符，则会报错。

## 2024.8.25

### Bug

将上一个版本的代码 pull 下来之后运行 `MyTensor.py`,得到报错信息如下：

```
Traceback (most recent call last):
  File "/home/petrichor/code/myTorch/MyTensor.py", line 437, in <module>
    z.backward()
  File "/home/petrichor/code/myTorch/MyTensor.py", line 228, in backward
    op.op_backward()
  File "/home/petrichor/code/myTorch/MyTensor.py", line 290, in op_backward
    add_grad=self.grad_func(node,self.output.grad)
  File "/home/petrichor/code/myTorch/MyTensor.py", line 378, in grad_func
    grad@self.last[1].data.swapaxes(-1,-2)
numpy.AxisError: axis2: axis -2 is out of bounds for array of dimension 1
```

## 2024.9.8 

### Bug

尝试用 MLP 跑了一下 MNIST 数据集的训练

```
[1,     1]: loss: 92.804 , acc: 9.38 %
/home/petrichor/code/myTorch/test/../MyTensor.py:399: RuntimeWarning: overflow encountered in matmul
  result=np.matmul(args[0].data,args[1].data)
/home/petrichor/code/myTorch/test/../MyTensor.py:399: RuntimeWarning: invalid value encountered in matmul
  result=np.matmul(args[0].data,args[1].data)
[1,   101]: loss: nan , acc: 9.75 %
```

感觉是 float 精度的问题。

并且现在的 MLP 层数好像是固定的，后续可以考虑用基类的方法自定义 MLP。

### BugFix

初始化问题，原来的初始化策略是(0,1)正态分布没有考虑fan_in 

后续修改并添加几个初始化策略后没什么问题了