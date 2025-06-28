import pytest
import numpy as np
import torch
from MyTorch_v2.myTensor import MyTensor
# from MyTorch_v2.my_nn import MLP
from MyTorch_v2.optim import BGD
from MyTorch_v2.loss_func import MSELoss
from MyTorch_v2.Dataloader import mydataLoader

# 测试 MyTensor 的基本运算
def test_mytensor_operations():
    # 创建 MyTensor 和 PyTorch 张量
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    my_tensor1 = MyTensor(data, requires_grad=True)
    my_tensor = MyTensor(data, requires_grad=True)
    torch_tensor = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    # 测试加法
    my_result = my_tensor + my_tensor1
    torch_result = torch_tensor + torch_tensor
    assert np.allclose(my_result.data, torch_result.detach().numpy()), "MyTensor addition failed"

    # 测试乘法
    my_result = my_tensor * 2
    torch_result = torch_tensor * 2
    assert np.allclose(my_result.data, torch_result.detach().numpy()), "MyTensor multiplication failed"

    # 测试梯度
    my_result = (my_tensor * my_tensor).sum()
    my_result.backward()
    torch_result = (torch_tensor * torch_tensor).sum()
    torch_result.backward()
    assert np.allclose(my_tensor.grad, torch_tensor.grad.numpy()), "MyTensor gradient computation failed"

# 测试 MLP 模型的前向传播
def test_mlp_forward():
    # 初始化数据
    input_data = np.random.randn(2, 2)
    my_tensor = MyTensor(input_data, requires_grad=True)
    torch_tensor = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)

    # 初始化 MyTorch 和 PyTorch 的 MLP 模型
    my_model = MLP(input_size=2, hidden_size=3, output_size=1, initial_policy='zeros')
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1, bias=True)
    )
    # 将 PyTorch 模型权重设置为零以匹配 initial_policy='zeros'
    with torch.no_grad():
        for param in torch_model.parameters():
            param.zero_()

    # 前向传播
    my_output = my_model.forward(my_tensor)
    torch_output = torch_model(torch_tensor)

    assert my_output.shape == (2, 1), "MyTensor MLP output shape incorrect"
    assert np.allclose(my_output.data, torch_output.detach().numpy(), atol=1e-5), "MLP forward pass mismatch with PyTorch"

# 测试 MSELoss
def test_mse_loss():
    # 准备数据
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[1.5, 2.5], [3.5, 4.5]])
    my_pred = MyTensor(pred, requires_grad=True)
    my_target = MyTensor(target)
    torch_pred = torch.tensor(pred, dtype=torch.float32, requires_grad=True)
    torch_target = torch.tensor(target, dtype=torch.float32)

    # 计算损失
    my_loss = MSELoss().forward(my_pred, my_target)
    torch_loss = torch.nn.MSELoss()(torch_pred, torch_target)

    assert np.allclose(my_loss.data, torch_loss.detach().numpy(), atol=1e-5), "MSELoss value mismatch"

    # 测试梯度
    my_loss.backward()
    torch_loss.backward()
    assert np.allclose(my_pred.grad, torch_pred.grad.numpy(), atol=1e-5), "MSELoss gradient mismatch"

# 测试 BGD 优化器
def test_bgd_optimizer():
    # 准备简单模型和数据
    my_model = MLP(2, 3, 1, initial_policy='zeros')
    torch_model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        for param in torch_model.parameters():
            param.zero_()

    # 准备输入和目标
    input_data = np.random.randn(2, 2)
    target_data = np.random.randn(2, 1)
    my_input = MyTensor(input_data, requires_grad=True)
    my_target = MyTensor(target_data)
    torch_input = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
    torch_target = torch.tensor(target_data, dtype=torch.float32)

    # 优化器
    my_optimizer = BGD(my_model.parameters, lr=0.01)
    torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)

    # 前向传播和损失
    my_output = my_model.forward(my_input)
    my_loss = MSELoss().forward(my_output, my_target)
    torch_output = torch_model(torch_input)
    torch_loss = torch.nn.MSELoss()(torch_output, torch_target)

    # 反向传播和优化
    my_optimizer.zero_grad()
    my_loss.backward()
    my_optimizer.step()

    torch_optimizer.zero_grad()
    torch_loss.backward()
    torch_optimizer.step()

    # 检查参数更新
    for my_param, torch_param in zip(my_model.parameters, torch_model.parameters()):
        assert np.allclose(my_param.data, torch_param.detach().numpy(), atol=1e-5), "BGD optimizer update mismatch"

# 测试 DataLoader
def test_dataloader():
    # 准备数据
    X = MyTensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    y = MyTensor(np.array([[0], [1], [1], [0]]))
    batch_size = 2

    # 创建 DataLoader
    dataloader = mydataLoader(X, y, batch_size=batch_size, shuffle=False, drop_last=False)

    # 检查批次数量
    assert len(dataloader) == 2, "DataLoader batch count incorrect"

    # 检查批次数据
    for i, (X_batch, y_batch) in enumerate(dataloader):
        assert X_batch.shape == (batch_size, 2), f"Batch {i} input shape incorrect"
        assert y_batch.shape == (batch_size, 1), f"Batch {i} target shape incorrect"
        assert np.allclose(X_batch.data, X.data[i * batch_size:(i + 1) * batch_size]), f"Batch {i} input data mismatch"
        assert np.allclose(y_batch.data, y.data[i * batch_size:(i + 1) * batch_size]), f"Batch {i} target data mismatch"

# 测试训练过程
def test_train_mytensor():
    # 准备数据
    X_train = MyTensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad=True)
    y_train = MyTensor(np.array([[0], [1], [1], [0]]), requires_grad=True)

    # 初始化模型和优化器
    model = MLP(2, 3, 1, initial_policy='zeros')
    optimizer = BGD(model.parameters, lr=0.01)
    dataloader = mydataLoader(X_train, y_train, batch_size=2, shuffle=False, drop_last=False)

    # 训练一个 epoch
    initial_loss = None
    for X_batch, y_batch in dataloader:
        y_pred = model.forward(MyTensor(X_batch))
        loss = MSELoss().forward(y_pred, MyTensor(y_batch))
        if initial_loss is None:
            initial_loss = loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 再次计算损失，检查是否减少
    final_loss = None
    for X_batch, y_batch in dataloader:
        y_pred = model.forward(MyTensor(X_batch))
        final_loss = MSELoss().forward(y_pred, MyTensor(y_batch)).data
        break

    assert final_loss < initial_loss, "Training did not reduce loss"