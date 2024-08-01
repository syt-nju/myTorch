#存储一些自己的常用工具函数
import os
import numpy as np
import matplotlib.pyplot as plt

def myAssert(judge: bool, message: str, *args, **kwargs):
    '''
    自定义断言函数
    judge:断言条件
    message:断言失败时的提示信息
    args:断言失败时的输出信息,可以是多个参数,最终会按顺序先输出参数的name属性(如果有),再输出参数的data属性
    '''
    if not judge:
        for i in args:
            if hasattr(i, 'name'):
                print(i.name,i.data)
            else:
                print(i)
        assert judge, message
def random_perturbation(array, perturbation_range, probability):
    """
    对一维NumPy数组进行随机扰动，使得每一项都有一定概率变成给定范围的另外一个整数

    :param array: 需要扰动的一维NumPy数组
    :param perturbation_range: 扰动范围，元组形式（最小值，最大值）  注意这里最大值取不到
    :param probability: 扰动的概率，范围 [0, 1]
    :return: 扰动后的NumPy数组
    """
    perturbed_array = array.copy()
    for i in range(len(array)):
        if np.random.rand() < probability:
            perturbed_array[i] = np.random.randint(perturbation_range[0], perturbation_range[1])
    return perturbed_array
if __name__ == "__main__":
    # #测试自定义断言函数
    # a=1
    # b=2
    # myAssert(a==b,f"{a}!={b}")
    # print("test passed")
    
    #测试数组扰动函数
    a=np.array([1,2,3])
    for i in range(10):
        a=random_perturbation(a,(1,4),1)
        print(a)