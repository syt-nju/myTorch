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

### 线性层及优化器等

基于矩阵乘法算子构成的线性层

```
    #测试BGD
    # #构造y=3x+2的模拟
    x=np.array(range(10)).reshape(-1,1)
    y_true=3*x+2+np.random.randn(10,1)*0.001
    layer=my_nn.MyLinearLayer(1,1,initial_policy='zeros')
    optimizer=BGD([layer.weight,layer.bias],lr=0.01)
    for i in range(1000):
        y_pred=layer.forward(MyTensor.MyTensor(x))
        
        loss=MSELoss().forward(y_pred,MyTensor.MyTensor(y_true))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(layer.weight.data,layer.bias.data)
```

输出

```
[[3.00085765]] [1.9946551]
```



## myTorch_v2

构建nn的时候被初版，繁琐的算子实例化流程过于恶心

修改架构为

​	1.计算图只存tensor，在init时决定是否进入tensor(更贴近pytorch逻辑)

​	2.把op的方法用类方法来使用，无需反复实例化，仅通过tensor.father_op即可索引，这样类方法就无需参与计算图和维护。

