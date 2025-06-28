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

## batchnorm 和 layernorm 的作用和区别

为什么要做数据归一化？

>保证所有特征在同一刻度下进行传递和比较
>
>如果不做归一化，数值较大的特征会在等weight下会计算出更大的结果，**淹没**其它特征带来的影响。
>
>在随机梯度下降的时候，假设y=aw+bw+cw, 如果a特别大，会导致梯度基本上由**a**决定，导致在**其它方向**上的优化极慢。(如图，像只有一边及其陡峭的峡谷)

![image-20241119004531218](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202411190045271.png)

batch norm在干一件什么事

>batch norm 在把一个batch的数据归一化(均值和方差由batch的值求出并进行动量式更新，并非人为规定)，再通过可以学习的超参数进行放缩移动到任意均值、方差的分布

![image-20241119005111755](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202411190051954.png)

> batchnorm和layernorm做的东西相似，batch norm是一个特征一个特征归一化，layernorm是对一个样本一个样本归一化，核心原因是在nlp领域，样本经常是个sequence，长度不定，此时对batch做normalization会导致很多比较后面的token跟一堆0一起算，显然会出问题，于是采用layernorm，对每个样本进行归一化。

## 为什么numpy 比 list快

> list 慢是显然的，他不是连续内存且存的东西不固定类型，批量处理肯定慢
>
> numpy有几个优势：
>
> ​	1.底层实现为C等更机器级别的语言，本身编译速度快
>
> ​	2.为同类型集中存储，适合一坨一次性搬进**缓存**，加快访问速度
>
> ​    3.向量化操作：对于向量的计算有着良好的可**并行**特性，numpy实现上做了并行化，充分利用CPU资源

## 为什么市面上的LLM 主流为decoder-only，而不是encoder-only 或者 encoder-decoder?

![image-20250610220853929](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202506102209020.png)

- decoder-only的三角矩阵必定满秩

  > 观点搬运自苏神(苏剑林)
  >
  > 原文
  >
  > “众所周知，Attention矩阵一般是由一个低秩分解的矩阵加softmax而来，具体来说是一个$𝑛×𝑑$的矩阵与$𝑑×𝑛$的矩阵相乘后再加softmax（$n≫d$），这种形式的Attention的矩阵因为低秩问题而带来表达能力的下降，具体分析可以参考[《Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth》](https://papers.cool/arxiv/2103.03404)。而Decoder-only架构的Attention矩阵是一个下三角阵，注意三角阵的行列式等于它对角线元素之积，由于softmax的存在，**对角线必然都是正数**，所以它的行列式必然是正数，即Decoder-only架构的**Attention矩阵一定是满秩**的！满秩意味着理论上有更强的表达能力，也就是说，Decoder-only架构的Attention矩阵在理论上具有更强的表达能力，改为双向注意力反而会变得不足。”
  >
  > 其实意思就是双向注意力可能面临训练过后矩阵rank坍塌的问题导致等参数下，模型的学习能力更差的问题

- decoder 任务天生难于encoder 任务，因此可能学到更好的表示

> Ilya 演讲中似乎提过这一点

## 能讲一下上层python写一个加法dispatch算子到整个流程吗



## 讲讲怎么判断两个tensor指向同一块现存



## autograd engine 对于分布式的情况是如何处理的
