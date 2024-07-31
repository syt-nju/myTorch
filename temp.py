#临时文件，不作任何用途，核心用途可能是拿来访问torch代码
#导入 pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#构造一个二维np
a = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])
#转换为torch
b = torch.tensor(a)
print(torch.softmax(b,dim=0))
