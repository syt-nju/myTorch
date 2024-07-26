'''实现所需要的算子'''

import numpy as np
import MyTensor
def sigmoid(x:MyTensor)->MyTensor:
    '''
    实现sigmoid函数
    '''
    x.data = 1/(1+np.exp(-x.data))


if __name__ == "__main__":
    a = MyTensor.MyTensor(np.array([0,2,3,4]))
    sigmoid(a)
    print(a)