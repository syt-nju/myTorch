import numpy as np

# 创建一个一维数组
arr_1d = np.array([1, 2, 3])

# 创建一个零维数组
arr_0d = np.array(5)

# 使用 atleast_2d 函数
arr_2d_from_1d = np.atleast_2d(arr_1d)
arr_2d_from_0d = np.atleast_2d(arr_0d)

print(arr_2d_from_1d)
print(arr_2d_from_1d.shape)
print(arr_2d_from_0d)
print(arr_2d_from_0d.shape)
print(arr_1d)