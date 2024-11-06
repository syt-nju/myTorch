"""
Modified by SkywardSigil
Date: 2024-09-22 22:11:00
Description: 
Add CrossEntropyLoss
"""


#实现损失函数，forward基于已经实现的func.py的函数
from typing import Tuple
from MyTorch.myTensor import MyTensor,Op,ComputationalGraph
from MyTorch import func
import numpy as np
class LossFunc(Op):
    '''
        损失函数类的基类
        继承自算子，但是要求不可调用forward(unsafe,如果这样要求每次调用都实例化来清理self.last)，
        请调用call方法
    '''
    #重载op_backward
    def op_backward(self):
        '''
        单个算子的反向传播
        '''
        node =self.last[0]#y_pred
        #由于loss函数只有一个输入，没有上级传入梯度等参数，不会出现广播
        node.grad+=self.grad_func()
    def forward(self,y_pred:MyTensor,y_true:MyTensor)->MyTensor:
        '''
        实现损失函数的前向传播
        @param y_pred:模型的输出
        @param y_true:真实标签
        @return:结果的MyTensor形式
        '''
        NotImplementedError
    def __call__(self, *args, **kwds):
        self.last = []#清空上次的记录,以后就不用反复实例化了
        return self.forward(*args, **kwds)
       
class MSELoss(LossFunc):
    def forward(self,y_pred:MyTensor, y_true:MyTensor, reduction = 'mean')->MyTensor:
        '''
        实现均方误差损失函数(MSE)
        @return:MyTensor 结果的MyTensor形式
        '''
        if isinstance(y_true,np.ndarray):
            y_true=MyTensor(y_true,requires_grad=False)
        #检查
        assert y_pred.shape == y_true.shape,'shape not match'
        assert y_pred.requires_grad==True,'y_pred must requires_grad'
        assert y_true.requires_grad==False,'y_true must not requires_grad'
        
        #记录此参数用于求grad_func时分类讨论
        self.reduction=reduction
        
        result=func.MSELoss(y_pred,y_true,reduction)
        ComputationalGraph.add_node(self)
        result.father_Op = self
        self.last=[y_pred,y_true]#顺序严格要求
        self.output=result
        return result
    def grad_func(self)->np.ndarray:
        '''
        计算梯度，仅返回值，不进行bp
        '''
        if self.reduction == 'mean':
            return 2*(self.last[0].data-self.last[1].data)/self.last[0].data.size
        elif self.reduction == 'sum':
            return 2*(self.last[0].data-self.last[1].data)
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
        
class CrossEntropyLoss(LossFunc):
    def forward(self, y_pred: MyTensor, y_true: MyTensor, reduction='mean') -> MyTensor:
        '''
        实现交叉熵损失函数
        @param y_pred: 模型的输出（未经过Softmax，直接传logits）
        @param y_true: 真实标签，one-hot或索引表示
        @param reduction: 选择返回损失是取平均还是求和
        @return: 结果的 MyTensor 形式
        '''
        if isinstance(y_true, np.ndarray):
            y_true = MyTensor(y_true, requires_grad=False)
        
        # 检查形状是否匹配
        assert y_pred.shape[0] == y_true.shape[0], 'shape not match'
        # 如果 y_true 是标签索引而非one-hot，需要转换成one-hot
        if len(y_true.shape) == 1:
            print("y_true is not one-hot, converting to one-hot...")
            one_hot_y_true = np.zeros_like(y_pred.data)
            one_hot_y_true[np.arange(y_true.shape[0]), y_true.data.astype(int)] = 1
            y_true = MyTensor(one_hot_y_true, requires_grad=False)

        self.reduction = reduction
        # result = func.CrossEntropyLoss(y_pred, y_true, reduction)
        # 因对于func.CrossEntropyLoss中的计算方法还有疑问因此暂时将这个修改后的实现直接放在这里
        # 计算 log(softmax)
        max_logit = np.max(y_pred.data, axis=1, keepdims=True)
        y_pred_sub_max = y_pred.data - max_logit
        log_sum_exp = np.log(np.sum(np.exp(y_pred_sub_max), axis=1, keepdims=True))
        simplified_cross = log_sum_exp - y_pred_sub_max
        
        # 交叉熵损失 = -真实标签的one-hot编码 * log(softmax)
        cross_entropy = np.sum(y_true.data * simplified_cross, axis=1)
        
        if reduction == 'mean':
            cross_entropy = np.mean(cross_entropy)
        elif reduction == 'sum':
            cross_entropy = np.sum(cross_entropy)
        
        # 创建 MyTensor 结果
        result = MyTensor(cross_entropy, requires_grad=True)


        ComputationalGraph.add_node(self)
        result.father_Op = self
        self.last = [y_pred, y_true]
        self.output = result
        
        return result
    
    def grad_func(self) -> np.ndarray:
        '''
        计算交叉熵损失的梯度
        '''
        # 计算 Softmax
        y_pred = self.last[0]
        y_true = self.last[1]

        grad = np.exp(y_pred.data - np.max(y_pred.data, axis=1, keepdims=True))  # Softmax
        grad /= np.sum(grad, axis=1, keepdims=True)  # Normalize to get probabilities
        
        # 交叉熵损失的梯度为 softmax 结果减去 one-hot 编码的真实标签
        grad -= y_true.data
        
        # 根据 reduction 参数选择返回平均梯度还是总和梯度
        if self.reduction == 'mean':
            grad /= y_true.data.shape[0]
        
        return grad

        
if __name__ == "__main__":
    # #简单测试
    # y_pred = MyTensor(np.array([1,2,3,3]),requires_grad=True)
    # y_true = np.array([1,2,3,4])
    # loss=MSELoss()
    # result=loss.forward(y_pred,y_true)
    # result.backward()
    # print(y_pred.grad)
    
    #基于pytorch对比的测试
    import torch
    import torch.nn as nn
    import my_nn
    # x=np.array(range(10)).reshape(-1,1)
    # y_true=3*x+2#+np.random.randn(10,1)*0.001
    # layer=my_nn.MyLinearLayer(1,1,initial_policy='zeros')
    # y_pred=layer.forward(MyTensor(x,requires_grad=True))
    # loss=MSELoss().forward(y_pred,MyTensor(y_true,requires_grad=False))
    # loss.backward()
    # print("grad",layer.weight.grad,layer.bias.grad)
    np.random.seed(0)
    torch.manual_seed(0)

    # 记录差异
    loss_diffs = []
    grad_diffs = []

    for i in range(10):
        logits = np.random.randn(2, 3).astype(np.float32)  # 随机生成logits
        labels = np.random.randint(0, 3, size=(2,), dtype=np.int32)  # 随机生成labels

        # MyTorch
        y_pred = MyTensor(logits, requires_grad=True)
        y_true = MyTensor(labels, requires_grad=False)

        my_loss = CrossEntropyLoss()
        result = my_loss.forward(y_pred, y_true, reduction='mean')
        result.backward()

        # PyTorch
        torch_logits = torch.tensor(logits, requires_grad=True)
        torch_labels = torch.tensor(labels, dtype=torch.long)

        torch_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        torch_loss = torch_loss_fn(torch_logits, torch_labels)
        torch_loss.backward()

        # 记录损失和梯度差异
        loss_diff = np.abs(result.data - torch_loss.item())
        grad_diff = np.abs(y_pred.grad - torch_logits.grad.detach().numpy()).max()

        loss_diffs.append(loss_diff)
        grad_diffs.append(grad_diff)

        # 输出每次迭代的结果
        print(f"Iteration {i+1}:")
        print("MyTorch CrossEntropyLoss Result:", result.data)
        print("PyTorch CrossEntropyLoss Result:", torch_loss.item())
        print("Loss Difference:", loss_diff)
        print("Gradient Difference:", grad_diff)
        print("-" * 40)

    # 统计输出
    print("Loss Differences (mean, std):", np.mean(loss_diffs), np.std(loss_diffs))
    print("Gradient Differences (mean, std):", np.mean(grad_diffs), np.std(grad_diffs))

    torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(labels, dtype=torch.long), reduction='mean')
