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

## 比较 pretraining, CPT（continual pre-training）,SFT,post training 

> 我们从训练数据、训练方式(损失函数)、训练目标三个角度来阐述这个问题。
>
> pre-trainning 使用自回归的训练方式，数据可以是爬下来的任何文本，本质上是教模型基础的语言逻辑和通用知识。
>
> CPT也叫pre-training 所以训练方式本质上还是pre-trainning，目的是让模型学会新知识或者是一些领域知识，为防止模型遗忘通用知识，一般把新数据和普通文本混合起来训练模型。
>
> SFT(supervised fine-tuning)则是让模型**学会对话**(或者类o1输出之类的要求)，一般用ChatTemplate去构造数据，一般还是自回归的训练方式。
>
> post-training 一般为RLHF等基于**强化学习**的对齐范式，包括PPO，DPO等微调，目的是对模型的回复引入人的偏好和安全性对齐等。一般认为post-training对模型的**修改量比较小**。
>
> 其实后面两种本质上都是微调，但是可以认为sft对模型的修改大于post-training，因此需要注意SFT的数据质量问题。

## 从深度学习的视角对李沐的参数服务器内容进行概述，构建一个基础的深度学习并行框架

### 所需要解决的问题

1.能在考虑性能的前提下，把分布式框架用于深度学习pipeline

2.构建具有一定可扩展性和容灾能力的分布式框架

### 框架

![image-20250109142456254](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202501091425387.png)

#### 可扩展性

为了可扩展性，李沐提出的框架以parameter server为核心，存权重，负责数据管理。而worker只是个打工的，负责计算或某种任务。由于**worker完全与存储剥离**，可扩展性拉满了。

(数据的分发可以额外做一个分发数据的管理器，也可以让参数服务器代劳，感觉不是很重要，因为没有同步需求咋滴都行)

#### 容灾方面

为减小通信成本，worker向参数服务器申请模型权重等数据的时候，通过index来申请。

意即，事先给weight做个**hash mapping**，key用一定范围的int来表示，相当于给了weight一些下标。用这些index来对weight进行指代，在不利用具体值的时候通过对index的通讯来传递信息。

而对权重的容灾就是**维护这个dict**。

![image-20250109144530040](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202501091445096.png)

具体来说，根据参数服务器的服务器数量，根据key的范围各自维护一段dict，而每个服务器又根据key的范围顺带备份后面两段的dict，相当于**循环备份了两份**，并且这些服务器本身就参与架构，可以瞬间启动。

### 通信成本问题

这套架构最大的问题就是通讯成本极高，如果不做调整，pull和push都得传一遍整个权重/整个梯度，相当炸裂。

为此，李沐在通讯前后对数据进行压缩，一般权重的压缩效果比较好，key就不太能压缩。

但是其实还是改变不了通讯成本太炸裂的问题。

### waiting time问题

由于不同worker执行完任务的时间是不一致的，为了减少worker间的相互等待，实操上我们选择**牺牲收敛速度**，不即时更新权重(有限度**延迟更新**或者完全延迟更新)

![image-20250109145702160](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202501091457207.png)

## 请介绍Megatron

> Megatron 是针对**大语言模型**的基于pytorch的**分布式**训练框架。
>
> 为解决模型参数过多的问题，设计了**三层并行**。
>
> 注意，我们看并行模型要注重看不同通信模式下的通信量。
>
> - **数据并行 (DP)**: 数据并行模式会在每个worker之上**复制一份模型**，这样每个worker都有一个完整模型的副本。输入数据集是分片的，一个训练的小批量数据将在多个worker之间分割；worker**定期汇总**它们的梯度，以确保所有worker看到一个一致的权重版本
> - **张量并行 (TP)**: 矩阵太大，显存太小放不下，只能把**矩阵运算分配到不同的设备之上**，比如把某个矩阵乘法切分成为多个矩阵乘法放到不同设备之上(对应的batch的矩阵也会被拆分)。
> - **流水线并行 (PP)**: 把模型**不同的层放到不同设备之上**，比如前面几层放到一个设备之上，中间几层放到另外一个设备上，最后几层放到第三个设备之上。（这样把pipeline进行拆分细化能够减小backward和forward的依赖性带来的waiting time）
>
> 

![image-20241224202714231](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202412242027330.png)

![image-20241224202736900](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202412242027981.png)

## 请给出以下代码的输出

```python
def get_iter(id):
    if id==1:
        yield from range(2)
    elif id==2:
        yield from range(3,5)
    else:
        return iter(range(10))
for i in range(3):
    for j in get_iter(i):
        print(j)

```

> 关键注意这个畜生 return， 在函数里面有yeild的情况下，return等价于StopIteration,输出如下

>0
>1
>3
>4

## 介绍deepspeed的ZeRO

> 为了减少训练阶段的显存消耗，deepspeed框架提供了ZeRO(Zero Redundancy Optimizer)方法。
>
> 回忆DP参数服务器，每个worker都得跑整个模型，存储对应的参数和，优化器参数等。
>
> 但是实际上都只需要一份就行了。
>
> ZeRO的思想就是去把这些进行聚合和切片，分到每个worker上。
>
> 分三个stage

![image-20250120150954183](https://typorasyt.oss-cn-nanjing.aliyuncs.com/202501201510308.png)

> 开的约高越省显存，但是由于这些资源每次调用都得worker之间通讯，**通讯成本大大增加**。
>
> 可以理解为，把整个模型拆分成了若干个层(与模型无关)，每次开始计算这个层的时候，左右worker经过通讯拿到所需要的该层相关的参数，结束后又广播或者求和后再广播，使得每个worker只存其中一部分参数。

> torch也对这个做了适配，叫做fsdp，实现上有一定差别，但是效果都是减少显存消耗。
>
> 调用方式上，都可以采用accelerate config 来进行配置后调用。代码中只需适配accelerate即可。
>
> 具体适配上的细节可以见网址：[7_accelerate (huaxiaozhuan.com)](https://www.huaxiaozhuan.com/工具/huggingface_transformer/chapters/7_accelerate.html)
