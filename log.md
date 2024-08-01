# debug 日志
## 2024.8.1
func.py中的nll函数出问题
测试函数运行结果：

```
(pytor-basic) PS C:\Users\nju22\Desktop\myTorch> & C:/conda/envs/pytor-basic/python.exe c:/Users/nju22/Desktop/myTorch/func.py
单输入检查完成
-1.3862943611198906     #你写的nll的输出
-1.0					#torch的nll的输出
[2 1 1]  				
[2 1 1]					#y_true和y_pred的值，详情请见myAssert的调用处
Traceback (most recent call last):
  File "c:\Users\nju22\Desktop\myTorch\func.py", line 148, in <module>
    myAssert(judge, f"{func.__name__} function failed in round {i}", result, result_torch.numpy(),y_true,y_pred)
  File "c:\Users\nju22\Desktop\myTorch\utils.py", line 19, in myAssert
    assert judge, message
AssertionError: NLL_loss function failed in round 0
```

