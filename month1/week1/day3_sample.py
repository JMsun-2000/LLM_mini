import torch

#如果 y 是向量 / 矩阵，就要给 backward() 传一个梯度权重，叫 grad_tensors。
x = torch.tensor([1., 2.], requires_grad=True)
y = x ** 2      # y = [1,4]

# 非标量必须传 grad_tensor
y.backward(gradient=torch.tensor([1,1]))
print(x.grad)   # d(y1+y2)/dx = [2,4]


#拓展 1：retain_graph（多次反向传播）
#如果想对同一个计算图多次 backward，需要加 retain_graph=True：
x = torch.tensor(2.0, requires_grad=True)
y = x**2
z = 3*y

# 第一次反向传播（保留计算图）
z.backward(retain_graph=True)
print(x.grad)  # dz/dx=6x=12 → tensor(12.)

# 清空梯度后，第二次反向传播
x.grad.zero_()
y.backward()
print(x.grad)  # dy/dx=2x=4 → tensor(4.)


#如果想让某个张量不参与梯度计算，用 .detach()：
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y_detach = y.detach()  # y_detach 不再追踪梯度
z = y_detach + 3

z.backward()  # 报错！因为y_detach切断了和x的梯度关联，计算图不完整