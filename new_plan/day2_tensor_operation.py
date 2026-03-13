import torch

# ====================== 一、张量基本运算（GPT计算的基础）======================
print("===== 1. 张量基本运算 =====")
# 1. 创建两个测试张量（后续GPT的Attention计算全是这类张量操作）
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 2. 基础算术运算（对应元素）
add = a + b  # 加法
sub = a - b  # 减法
mul = a * b  # 乘法（对应元素相乘，GPT里叫element-wise）
div = a / b  # 除法
print("加法:\n", add)
print("减法:\n", sub)
print("对应元素乘法:\n", mul)
print("除法:\n", div)

# 3. 矩阵乘法（GPT的Attention核心运算！重点记 @ 或 mm）
matmul1 = a @ b       # 推荐写法，直观
matmul2 = torch.mm(a, b)  # 等价写法
print("矩阵乘法:\n", matmul1)

# ====================== 二、张量索引与切片（你重点要补的内容）======================
print("\n===== 2. 张量索引与切片 =====")
# 创建一个3x4的测试张量（模拟GPT的输入序列张量）
x = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

# 1. 基础索引（行/列）：GPT里取某一段序列会用
print("原始张量x:\n", x)
# 取第0行（GPT里取第0个token的特征）
row_0 = x[0]
print("第0行:", row_0)
# 取第1行第2列（单个元素）
elem_1_2 = x[1, 2]
print("第1行第2列:", elem_1_2)
# 取所有行的第3列（GPT里取所有token的某个特征维度）
col_3 = x[:, 3]
print("所有行的第3列:", col_3)

# 2. 切片（取范围）：GPT里切分上下文序列的核心操作
# 取第1-2行（不包含3）、第0-2列（不包含2）
slice_1 = x[1:3, 0:2]
print("第1-2行，第0-2列:\n", slice_1)
# 取所有行，前3列（GPT里取序列前n个token）
slice_2 = x[:, :3]
print("所有行，前3列:\n", slice_2)

# 3. 条件索引：GPT里过滤无效值（比如padding的0）
# 取x中大于50的所有元素
cond_idx = x[x > 50]
print("x中大于50的元素:", cond_idx)

# ====================== 三、形状操作（GPT调整张量维度的核心）======================
print("\n===== 3. 形状操作 =====")
# 创建一个测试张量（模拟GPT的[batch_size, seq_len, hidden_dim]）
y = torch.randn(2, 3, 4)  # 2个batch，3个token，4维特征
print("原始张量y形状:", y.shape)  # torch.Size([2, 3, 4])

# 1. reshape / view（改变形状，数据不变）：GPT里频繁调整维度
# 把2x3x4变成6x4（合并batch和seq_len）
y_reshaped = y.reshape(6, 4)
print("reshape(6,4)后的形状:", y_reshaped.shape)
# 把6x4变回2x3x4（GPT里恢复原始维度）
y_back = y_reshaped.view(2, 3, 4)
print("view(2,3,4)后的形状:", y_back.shape)

# 2. squeeze / unsqueeze（增/减维度）：GPT里加batch维度/减维度常用
# unsqueeze：给y加一个维度（比如在第0位加batch维度，从3x4→1x3x4）
z = torch.randn(3, 4)
z_unsq = z.unsqueeze(0)  # 加第0维
print("unsqueeze(0)后的形状:", z_unsq.shape)  # torch.Size([1, 3, 4])
# squeeze：去掉维度为1的维度（把1x3x4→3x4）
z_sq = z_unsq.squeeze(0)
print("squeeze(0)后的形状:", z_sq.shape)  # torch.Size([3, 4])

# 3. transpose / permute（维度交换）：GPT的Attention里转置Q/K/V用
# transpose：交换两个维度（比如把2x3x4的第1和第2维交换→2x4x3）
y_trans = y.transpose(1, 2)
print("transpose(1,2)后的形状:", y_trans.shape)
# permute：交换多个维度（比如把2x3x4→2x4x3，和上面等价）
y_perm = y.permute(0, 2, 1)
print("permute(0,2,1)后的形状:", y_perm.shape)

# ====================== 四、Day2 小练习（验证掌握程度）======================
print("\n===== 4. Day2 小练习 =====")
# 练习目标：把随机张量切片→变形→矩阵乘，跑通完整流程
# 1. 创建4x4随机张量
practice = torch.randint(0, 10, (4, 4))
print("练习原始张量:\n", practice)
# 2. 切片：取前3行，后2列
practice_slice = practice[:3, 2:]
print("切片后:\n", practice_slice)  # 形状3x2
# 3. 变形：把3x2变成2x3
practice_reshape = practice_slice.reshape(2, 3)
print("变形后:\n", practice_reshape)  # 形状2x3
# 4. 矩阵乘：和自己的转置相乘（2x3 × 3x2 = 2x2）
practice_matmul = practice_reshape @ practice_reshape.T
print("矩阵乘结果:\n", practice_matmul)
