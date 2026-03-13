import torch
import time

# 大张量运算，对比CPU和MPS速度
device_cpu = torch.device("cpu")
device_mps = torch.device("mps")

# 创建大张量（10000x10000）
a = torch.randn(10000, 10000)
b = torch.randn(10000, 10000)

# CPU运算时间
start = time.time()
a_cpu = a.to(device_cpu)
b_cpu = b.to(device_cpu)
c_cpu = a_cpu @ b_cpu
print(f"CPU矩阵乘耗时: {time.time()-start:.2f}秒")

# MPS运算时间（M系列Mac）
if torch.backends.mps.is_available():
    start = time.time()
    a_mps = a.to(device_mps)
    b_mps = b.to(device_mps)
    c_mps = a_mps @ b_mps
    # 等待MPS计算完成（异步执行）
    torch.mps.synchronize()
    print(f"MPS矩阵乘耗时: {time.time()-start:.2f}秒")