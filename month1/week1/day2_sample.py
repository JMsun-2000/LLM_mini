import torch

# 1. 开启梯度追踪的张量创建
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# 2. 构建计算图：z = x² + y + 3
z = x**2 + y + 3

# 3. 反向传播求梯度
z.backward()  # 计算dz/dx、dz/dy

# 4. 查看梯度
print(x.grad)  # 输出tensor([4.])，dz/dx=2x=4
print(y.grad)  # 输出tensor([1.])，dz/dy=1

# 5. 禁用梯度追踪的两种方式
## 方式1：torch.no_grad()上下文
with torch.no_grad():
    z_no_grad = x**2 + y
    print(z_no_grad.requires_grad)  # 输出False
## 方式2：detach()，创建无梯度的张量副本
x_detach = x.detach()
x_detach.requires_grad  # 输出False

# 6. 大模型常用：梯度清零（训练循环必备）
w = torch.randn(3, 1, requires_grad=True)
for i in range(2):
    loss = w.sum()
    loss.backward()
    print(w.grad)  # 第一次输出tensor([[1.],[1.],[1.]]), 第二次累加为[[2.],[2.],[2.]]
    w.grad.zero_()  # 梯度清零，训练循环中必须做！