# 1. 基础导入与环境检查
import torch
import numpy as np
# 检查PyTorch版本和CUDA是否可用
print(torch.__version__)
print(torch.cuda.is_available())  # 返回True则GPU可用

# 2. 张量的创建（5种最常用方式）
t1 = torch.tensor([[1,2,3], [4,5,6]])  # 从列表创建
t2 = torch.zeros(2,3)  # 全0张量
t3 = torch.ones(2,3, dtype=torch.float32)  # 全1张量，指定数据类型
t4 = torch.randn(2,3)  # 标准正态分布随机张量
t5 = torch.from_numpy(np.array([[1,2],[3,4]]))  # 从Numpy数组创建

# 3. 张量核心属性查看
print(t1.shape)  # 输出torch.Size([2, 3])，维度
print(t1.dtype)  # 输出torch.int64，默认数据类型
print(t1.device)  # 输出cpu，默认存储设备

# 4. 张量核心操作（大模型常用）
## 维度变换（reshape/permute，重点！）
t1_reshape = t1.reshape(3,2)  # 改变形状，不改变数据顺序
t1_permute = t1.permute(1,0)  # 维度置换，适合矩阵转置/高维张量
## 索引与切片（取指定数据）
t1_slice = t1[:, 1:]  # 取所有行，第2列及以后
## 数学运算（加减乘除+矩阵乘法，重点！）
t_mul = t1 * 2  # 逐元素乘法
t_matmul = t1 @ t1.permute(1,0)  # 矩阵乘法，大模型核心运算
## 设备迁移（CPU→GPU，重点！）
if torch.cuda.is_available():
    t1_cuda = t1.cuda()  # 迁移到GPU
    # 或 t1_cuda = t1.to('cuda')

# 5. 张量与Numpy互通
t_np = t1.numpy()  # 张量→Numpy
np_t = torch.from_numpy(t_np)  # Numpy→张量