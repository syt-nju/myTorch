import functools

grad_enabled = True

def is_grad_enabled():
    return grad_enabled

def set_grad_enabled(mode: bool):
    global grad_enabled
    grad_enabled = mode

class no_grad:
    '''
    一个上下文管理器和装饰器，用于在特定代码块或函数中禁用梯度计算。
    方法:
    -------
    __enter__():
        进入上下文管理器时，保存当前的梯度计算状态并禁用梯度计算。
    
    __exit__(exc_type, exc_value, traceback):
        退出上下文管理器时，恢复进入上下文管理器前的梯度计算状态。
    
    __call__(func):
        将该类用作装饰器时，装饰传入的函数，使其在禁用梯度计算的上下文中运行。
    '''

    def __enter__(self)->None:
        self.prev = is_grad_enabled()
        set_grad_enabled(False)
    
    def __exit__(self, exc_type, exc_value, traceback)->None:
        set_grad_enabled(self.prev)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

#TODO: 实现 enable_grad 装饰器,仿照 no_grad 即可，主要觉得可以通过这个理解一下装饰器的用法和实现
class enable_grad:

    NotImplementedError

'''
装饰器用法示例
'''

@no_grad()
def test_no_grad():
    print(is_grad_enabled())
    print("test no grad")

if __name__ == "__main__":
    test_no_grad()
    print(is_grad_enabled())
    print("end")