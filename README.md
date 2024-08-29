# myTorch

## 简介

参考学长的项目，基于numpy尝试对pytorch的功能进行复现。

## 已实现功能

### MyTensor

实现了MyTensor

可指定dtype，默认值为float

example：

```python
x=MyTensor(np.array([1,2,3]),requires_grad=False)
```

重载了运算符，但这些运算符不会参与grad计算

### Operator与backward

代码的backward采用对计算图中的operator进行遍历实现反向传播

故所有参与反向传播的operator均需要实例化

```python
    #实现y=ax+b
    x=MyTensor(np.array([1,2,3]),requires_grad=False)
    a=MyTensor(np.array([1,1,1]),requires_grad=True)
    b=MyTensor(np.array([7,8,9]),requires_grad=True)
    add=Sum()
    mul=Mul()
    temp=mul.forward(a,x)
    y=add.forward(temp,b)
    print(y)
    y.backward()
    print(a.grad,b.grad)
```

输出为

```shell
MyTensor([ 8. 10. 12.])
[1. 2. 3.] [1. 1. 1.]
```

### 线性层

基于矩阵乘法算子构成的线性层

```
    layer = MyLinearLayer(1,3,initial_policy='zeros')
    x = MyTensor([9],requires_grad=False)
    y = layer.forward(x)
    y.backward()
    print(layer.weight.grad)
    print(layer.bias.grad)
```

输出

```
[[9. 9. 9.]]
[1. 1. 1.]
```

