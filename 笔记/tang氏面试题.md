# tang氏面试题

## 为什么牛顿法不能作为优化器算法

相关链接：[牛顿法最优化介绍](https://zh.d2l.ai/chapter_optimization/gd.html#id7)

>1.牛顿法对于每个变量需要保存一个hessian矩阵，即f(x)的二阶信息，这个存储量过大(所以深度学习框架都是基于一阶信息进行优化)
>
>2.解析解包含了对矩阵的求逆，可能遇到非奇异矩阵



## 简述ada系列优化器的特点？如何实现的？

> ada系列优化器的特点为：根据参数历史梯度来决定该轮的更新量，具有自适应的特点，因此不怎么需要调节优化器相关参数。
>
> 最早的adagrad直接把所有的历史梯度平方和记录进来，因此随着训练轮次增加，很容易导致下降速度过慢甚至趋于0。
>
> 因此后续的adadelta考虑深度学习的价值函数的设计，通过y_t=k*y_{t-1}+(1-k)\*grad的设计使得远端的梯度会被忽略，解决速度下降过快的问题。
>
> Adam则是将ada的思路和动量的思路结合起来，通过adadelta的形式，记录近期梯度平方和和近期梯度和(动量)，通过一些简单工程上的矫正，得到最终的更新式。

## torch.nn.softmax和torch.nn.functional.softmax有什么区别？是否会对计算图产生影响？

> 前者是class，后者是function，调用方式不一样，但是都会对计算图产生影响，因为他都调用了会影响计算图的算子。

## 上下文窗口的限制的底层原因是什么？有什么解决方案吗？

> 自注意力机制需要对序列进行内积，时间空间复杂度都是O(n^2)，考虑显存大小和上下文窗口，序列长度会受到限制。
>
> 最常规的就是接入RAG，能够添加外部知识，并且扩大有效输入范围。
>
> 还有一个叫稀疏自注意力机制的东西，让token只关心前后的一定范围的token，在一个个小范围进行自注意力，就能降低复杂度。

## 请回答下图和注意力机制有什么区别，哪里不一样了？

![a9ad57ee8a098832e7dccd77fd957bc](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202410262136006.png)

> 这是group ViT论文中的group block，用于做软聚类
>
> 最大的差别是，注意力机制把value做了加权和，这里是保留了整个加权矩阵，可以看作向多个聚类中心聚类的结果。只需调节gumble softmax的温度，使得系数逼近 0/1 ，那么最终结果就是良好的软聚类，还维持了可导的性质
