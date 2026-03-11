import torch

# day1 homework
# 用 PyTorch 创建一个3×4×5的随机张量，修改其数据类型为torch.float16，并迁移到 GPU（若有）
t1 = torch.randn(3, 4, 5, dtype=torch.float16)
if torch.cuda.is_available():
    t1 = t1.cuda()

# 对上述张量做维度置换（将第 1 维和第 3 维交换），再reshape为6×10的张量
t1_permute = t1.permute(2, 1, 0)
t1_reshape = t1_permute.reshape(6, 10)

# 实现两个 2×3 张量的逐元素乘法和矩阵乘法（注意矩阵乘法的维度匹配要求）
t1 = torch.randn(2, 3)
t2 = torch.randn(2, 3)
t_mul = t1 * t2
t_matmul = t1 @ t2.T


# day2 homework
# 定义函数y = 3x³ + 2x² - 5x + 1，用 PyTorch 求 x=1 时的导数dy/dx
x = torch.tensor([1.0], requires_grad=True)
y = x**3 + 2*x**2 - 5*x + 1

y.backward()
print(x.grad)

# 创建一个 2×2 的随机权重张量w，构建损失函数loss = (w @ w).sum()，实现两次反向传播，并在每次传播前做梯度清零，观察梯度变化
w = torch.randn(2, 2, requires_grad=True)
for i in range(2):
    loss = (w @ w).sum()
    loss.backward()
    print(w.grad)
    w.grad.zero_()



# day3 homework
x = torch.tensor(3.0, requires_grad=True)
y = x**4 + 2*x**3 - x + 5
y.backward()
print(x.grad)

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = 3*x + 1
y.backward(torch.ones_like(y))
print(x.grad)

a = torch.tensor(2., requires_grad=True)
b = a**2
c = b + 3
d = 2*c
d.backward()
print(a.grad)


w = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
print(w*w)
loss1 = (w * w).sum()
print(loss1)
print(w @ w)
loss2 = (w @ w).sum()
print(loss2)
total_loss = 0.5*loss1 + 0.5*loss2
total_loss.backward()
print(w.grad)