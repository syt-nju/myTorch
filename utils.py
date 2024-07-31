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

if __name__ == "__main__":
    #测试自定义断言函数
    a=1
    b=2
    myAssert(a==b,f"{a}!={b}")
    print("test passed")