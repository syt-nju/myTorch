import pytest
import numpy as np
import torch
import torch.nn.functional as F
from MyTorch.myTensor import MyTensor, Sum, Sub, Mul, MatMul, Div, Max, Exp, Log, SumUnary
from MyTorch.my_nn import ReLU, Softmax, LogSoftmax

# 用于比较两个结果是否足够接近的辅助函数
def assert_close(a, b, rtol=1e-5, atol=1e-8):
    assert np.allclose(a, b, rtol=rtol, atol=atol), f"Arrays not close: \na={a}\nb={b}"

# 基本张量操作测试
def test_tensor_creation():
    data = np.random.randn(3, 4)
    my_tensor = MyTensor(data)
    torch_tensor = torch.tensor(data)
    
    assert_close(my_tensor.data, torch_tensor.numpy())

# 二元操作的参数化测试（包括前向传播和梯度计算）
@pytest.mark.parametrize(
    "my_op, torch_op, shape_a, shape_b", [
        (Sum.forward, lambda a, b: a + b, (3, 4), (3, 4)),
        (Sub.forward, lambda a, b: a - b, (3, 4), (3, 4)),
        (Mul.forward, lambda a, b: a * b, (3, 4), (3, 4)),
        (MatMul.forward, lambda a, b: a @ b, (3, 4), (4, 2)),
        (Div.forward, lambda a, b: a / b, (3, 4), (3, 4)),
    ]
)
def test_binary_ops(my_op, torch_op, shape_a, shape_b):
    a_data = np.random.randn(*shape_a)
    b_data = np.random.randn(*shape_b)
    
    # 对于除法，确保除数不接近零
    if my_op == Div.forward:
        b_data = np.where(np.abs(b_data) < 0.1, 0.1, b_data)
    
    # 测试前向传播
    my_a = MyTensor(a_data)
    my_b = MyTensor(b_data)
    torch_a = torch.tensor(a_data, dtype=torch.float32)
    torch_b = torch.tensor(b_data, dtype=torch.float32)
    
    my_result = my_op(my_a, my_b)
    torch_result = torch_op(torch_a, torch_b)
    
    assert_close(my_result.data, torch_result.numpy())
    
    # 测试梯度计算
    my_a_grad = MyTensor(a_data, requires_grad=True)
    my_b_grad = MyTensor(b_data, requires_grad=True)
    torch_a_grad = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
    torch_b_grad = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
    
    my_result_grad = my_op(my_a_grad, my_b_grad)
    torch_result_grad = torch_op(torch_a_grad, torch_b_grad)
    
    my_result_grad.backward()
    torch_result_grad.backward(torch.ones_like(torch_result_grad))
    
    assert_close(my_a_grad.grad, torch_a_grad.grad.numpy())
    assert_close(my_b_grad.grad, torch_b_grad.grad.numpy())

# 一元操作的参数化测试（包括前向传播和梯度计算）
@pytest.mark.parametrize(
    "my_op, torch_op, data_gen", [
        (Exp.forward, torch.exp, lambda: np.random.randn(3, 4)),
        (Log.forward, torch.log, lambda: np.random.rand(3, 4) + 0.1),  # 确保为正值
        (lambda x: ReLU.forward(x), F.relu, lambda: np.random.randn(3, 4)),
    ]
)
def test_unary_ops(my_op, torch_op, data_gen):
    data = data_gen()
    
    # 测试前向传播
    my_tensor = MyTensor(data)
    torch_tensor = torch.tensor(data, dtype=torch.float32)
    
    my_result = my_op(my_tensor)
    torch_result = torch_op(torch_tensor)
    
    assert_close(my_result.data, torch_result.numpy())
    
    # 测试梯度计算
    my_tensor_grad = MyTensor(data, requires_grad=True)
    torch_tensor_grad = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    
    my_result_grad = my_op(my_tensor_grad)
    torch_result_grad = torch_op(torch_tensor_grad)
    
    my_result_grad.backward()
    torch_result_grad.backward(torch.ones_like(torch_result_grad))
    
    assert_close(my_tensor_grad.grad, torch_tensor_grad.grad.numpy())

# 带轴参数的操作的参数化测试（包括前向传播和梯度计算）
@pytest.mark.parametrize(
    "my_op, torch_op, shape, axes", [
        (
            lambda x, axis, keepdims: Max.forward(x, axis=axis, keepdims=keepdims),
            lambda x, dim, keepdim: torch.max(x, dim=dim, keepdim=keepdim)[0],
            (3, 4, 5),
            [0, 1, 2]
        ),
        (
            lambda x, axis, keepdims: SumUnary.forward(x, axis=axis, keepdims=keepdims),
            lambda x, dim, keepdim: torch.sum(x, dim=dim, keepdim=keepdim),
            (3, 4, 5),
            [0, 1, 2]
        ),
    ]
)
def test_axis_ops(my_op, torch_op, shape, axes):
    data = np.random.randn(*shape)
    
    for axis in axes:
        # 测试前向传播 - 保留维度
        my_tensor = MyTensor(data)
        torch_tensor = torch.tensor(data, dtype=torch.float32)
        
        my_result = my_op(my_tensor, axis, True)
        torch_result = torch_op(torch_tensor, axis, True)
        assert_close(my_result.data, torch_result.numpy())
        
        # 测试前向传播 - 不保留维度
        my_result = my_op(my_tensor, axis, False)
        torch_result = torch_op(torch_tensor, axis, False)
        assert_close(my_result.data, torch_result.numpy())
        
        # 测试梯度计算 - 保留维度
        my_tensor_grad = MyTensor(data, requires_grad=True)
        torch_tensor_grad = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        
        my_result_grad = my_op(my_tensor_grad, axis, True)
        torch_result_grad = torch_op(torch_tensor_grad, axis, True)
        
        my_result_grad.backward()
        torch_result_grad.backward(torch.ones_like(torch_result_grad))
        
        assert_close(my_tensor_grad.grad, torch_tensor_grad.grad.numpy())
        
        # 重置梯度
        my_tensor_grad.zero_grad()
        torch_tensor_grad.grad.zero_()
        
        # 测试梯度计算 - 不保留维度
        my_result_grad = my_op(my_tensor_grad, axis, False)
        torch_result_grad = torch_op(torch_tensor_grad, axis, False)
        
        my_result_grad.backward()
        torch_result_grad.backward(torch.ones_like(torch_result_grad))
        
        assert_close(my_tensor_grad.grad, torch_tensor_grad.grad.numpy())

# 测试Softmax和LogSoftmax（包括前向传播和梯度计算）
@pytest.mark.parametrize(
    "my_class, torch_op, dims", [
        (Softmax, F.softmax, [0, 1]),
        (LogSoftmax, F.log_softmax, [0, 1]),
    ]
)
def test_softmax_ops(my_class, torch_op, dims):
    data = np.random.randn(3, 4)
    
    for dim in dims:
        # 测试前向传播
        my_tensor = MyTensor(data)
        torch_tensor = torch.tensor(data, dtype=torch.float32)
        
        my_op = my_class(dim=dim)
        my_result = my_op.forward(my_tensor)
        torch_result = torch_op(torch_tensor, dim=dim)
        
        assert_close(my_result.data, torch_result.numpy())
        
        # 测试梯度计算
        my_tensor_grad = MyTensor(data, requires_grad=True)
        torch_tensor_grad = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        
        my_result_grad = my_op.forward(my_tensor_grad)
        torch_result_grad = torch_op(torch_tensor_grad, dim=dim)
        
        my_result_grad.backward()
        torch_result_grad.backward(torch.ones_like(torch_result_grad))
        
        assert_close(my_tensor_grad.grad, torch_tensor_grad.grad.numpy())

# 复杂计算图的测试
def test_complex_computation_graph():
    # 创建输入数据
    a_data = np.random.randn(3, 4)
    b_data = np.random.randn(4, 5)
    c_data = np.random.randn(3, 5)
    
    # 测试计算图1: (a @ b) * c
    # MyTorch计算
    my_a = MyTensor(a_data, requires_grad=True)
    my_b = MyTensor(b_data, requires_grad=True)
    my_c = MyTensor(c_data, requires_grad=True)
    
    my_ab = MatMul.forward(my_a, my_b)
    my_result = Mul.forward(my_ab, my_c)
    my_result.backward()
    
    # PyTorch计算
    torch_a = torch.tensor(a_data, requires_grad=True)
    torch_b = torch.tensor(b_data, requires_grad=True)
    torch_c = torch.tensor(c_data, requires_grad=True)
    
    torch_ab = torch_a @ torch_b
    torch_result = torch_ab * torch_c
    torch_result.backward(torch.ones_like(torch_result))
    
    # 比较梯度
    assert_close(my_a.grad, torch_a.grad.numpy())
    assert_close(my_b.grad, torch_b.grad.numpy())
    assert_close(my_c.grad, torch_c.grad.numpy())
    
    # 测试计算图2: exp(a @ b) + c
    # 清除梯度
    my_a.zero_grad()
    my_b.zero_grad()
    my_c.zero_grad()
    torch_a.grad.zero_()
    torch_b.grad.zero_()
    torch_c.grad.zero_()
    
    # MyTorch计算
    my_ab = MatMul.forward(my_a, my_b)
    my_exp_ab = Exp.forward(my_ab)
    my_result = Sum.forward(my_exp_ab, my_c)
    my_result.backward()
    
    # PyTorch计算
    torch_ab = torch_a @ torch_b
    torch_exp_ab = torch.exp(torch_ab)
    torch_result = torch_exp_ab + torch_c
    torch_result.backward(torch.ones_like(torch_result))
    
    # 比较梯度
    assert_close(my_a.grad, torch_a.grad.numpy())
    assert_close(my_b.grad, torch_b.grad.numpy())
    assert_close(my_c.grad, torch_c.grad.numpy())
