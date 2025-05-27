from MyTorch_v2.myTensor import MyTensor,MyTensor,MatMul,Sum,Mul,Max,Exp,Log,Sub,SumUnary,Div
from MyTorch_v2.myTensor import ComputationalGraph,op_forward

import numpy as np
class MSELoss():
    def forward(self,x:MyTensor,y:MyTensor,**kwargs)->MyTensor:
        return SumUnary.forward((x-y)*(x-y),axis=None)
    def __call__(self,x:MyTensor,y:MyTensor)->MyTensor:
        return self.forward(x,y)
class CrossEntropyLoss():
    def forward(self,y_pred:MyTensor,y_true:MyTensor,**kwargs)->MyTensor:
        '''
        实现交叉熵损失函数
        @param y_pred: 模型的输出（未经过Softmax，直接传logits）
        @param y_true: 真实标签，one-hot或索引表示
        @return: 结果的 MyTensor 形式
        '''
        if len(y_true.shape) == 1:
            print("y_true is not one-hot, converting to one-hot...")
            one_hot_y_true = np.zeros_like(y_pred.data)
            for i in range(y_true.shape[0]):
                one_hot_y_true[i, y_true.data[i].astype(int)] = 1
            y_true = MyTensor(one_hot_y_true, requires_grad=False)
        # 计算 log(softmax)
        # max_logit = np.max(y_pred.data, axis=1, keepdims=True)
        # y_pred_sub_max = y_pred.data - max_logit
        # log_sum_exp = np.log(np.sum(np.exp(y_pred_sub_max), axis=1, keepdims=True))
        # simplified_cross = log_sum_exp - y_pred_sub_max
        max_logit = Max.forward(y_pred, axis=1, keepdims=True)
        y_pred_sub_max = y_pred - max_logit
        exp = Exp.forward(y_pred_sub_max)
        sum_exp = SumUnary.forward(exp, axis=1, keepdims=True)
        log_sum_exp = Log.forward(sum_exp)
        simplified_cross = log_sum_exp - y_pred_sub_max
        
        # 计算交叉熵
        # 交叉熵损失 = -真实标签的one-hot编码 * log(softmax)
        zeros=MyTensor(np.zeros_like(y_pred.data),requires_grad=False)
        temp=y_true*simplified_cross
        cross_entropy_loss=zeros-temp
        return SumUnary.forward(cross_entropy_loss,axis=None)
        
    def __call__(self,x:MyTensor,y:MyTensor)->MyTensor:
        return self.forward(x,y)
class NLLLoss():
    def forward(self,y_pred:MyTensor,y_true:MyTensor,**kwargs)->MyTensor:
        '''
        实现负对数似然损失函数
        @param y_pred: 模型的输出（经过logSoftmax）
        @param y_true: 真实标签，one-hot或索引表示
        @param reduction: 选择返回损失是取平均还是求和
        @return: 结果的 MyTensor 形式
        '''
        if len(y_true.shape) == 1:
            print("y_true is not one-hot, converting to one-hot...")
            one_hot_y_true = np.zeros_like(y_pred.data)
            for i in range(y_true.shape[0]):
                one_hot_y_true[i, y_true.data[i].astype(int)] = 1
            y_true = MyTensor(one_hot_y_true, requires_grad=False)
        simplified_cross = y_pred
        # 计算交叉熵
        # 交叉熵损失 = -真实标签的one-hot编码 * log(softmax)
        zeros=MyTensor(np.zeros_like(y_pred.data),requires_grad=False)
        temp=y_true*simplified_cross
        cross_entropy_loss=zeros-temp
        return SumUnary.forward(cross_entropy_loss,axis=None)
        
    def __call__(self,x:MyTensor,y:MyTensor)->MyTensor:
        return self.forward(x,y)
