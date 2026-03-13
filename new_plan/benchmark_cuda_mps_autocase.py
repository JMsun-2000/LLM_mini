import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast  # 注意：这个 autocast 也兼容 MPS


'''
补充：MPS 与 CUDA 混合精度训练的完整对比

为了让你更清晰地看到两者的完整流程差异，我整理了对比表格：

步骤        CUDA 混合精度	                        MPS 混合精度
梯度清零    optimizer.zero_grad()（必须）           optimizer.zero_grad()（必须）
前向传播	with autocast(device_type="cuda"):      with autocast(device_type="mps"):
反向传播	scaler.scale(loss).backward()	        loss.backward()（自动缩放）
参数更新	scaler.step(optimizer)	                optimizer.step()
缩放器更新	scaler.update()	                        无需此步骤
总结

梯度清零是通用要求：无论 CUDA/MPS，训练时都必须通过 optimizer.zero_grad() 清零梯度，这是保证训练正确的核心步骤。
CUDA 混合精度的核心差异：必须显式使用 GradScaler() 完成梯度缩放、参数更新和缩放器状态更新，而 MPS 无需手动处理缩放。
代码规范：训练循环的标准流程是「清零梯度 → 前向传播 → 计算损失 → 反向传播 → 更新参数」，混合精度仅改变后三步的实现形式，不改变整体流程。
'''

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 初始化模型、优化器、损失函数
model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# --------------------------
# MPS 混合精度训练核心代码
# --------------------------
# 注意：MPS 不需要创建 GradScaler()！
# PyTorch 会自动处理梯度缩放，避免数值下溢

# 模拟训练数据
batch_size = 32
x = torch.randn(batch_size, 100).to(device)
y = torch.randint(0, 10, (batch_size,)).to(device)

# 训练步骤
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    
    # 1. 启用 autocast 自动混合精度
    with autocast(device_type="mps", dtype=torch.float16):
        outputs = model(x)
        loss = criterion(outputs, y)
    
    # 2. 反向传播（MPS 自动处理梯度缩放）
    loss.backward()
    
    # 3. 优化器更新
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --------------------------
# 对比：CUDA 混合精度写法（仅供参考）
# --------------------------
if torch.cuda.is_available():
    device_cuda = torch.device("cuda")
    model_cuda = SimpleModel().to(device_cuda)
    optimizer_cuda = optim.Adam(model_cuda.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()  # CUDA 必须显式创建 GradScaler
    
    x_cuda = torch.randn(batch_size, 100).to(device_cuda)
    y_cuda = torch.randint(0, 10, (batch_size,)).to(device_cuda)
    
    model_cuda.train()
    with autocast(device_type="cuda", dtype=torch.float16):
        outputs_cuda = model_cuda(x_cuda)
        loss_cuda = criterion(outputs_cuda, y_cuda)
    
    scaler.scale(loss_cuda).backward()  # CUDA 需要用 scaler 缩放损失
    scaler.step(optimizer_cuda)         # CUDA 需要用 scaler 执行优化步骤
    scaler.update()                     # CUDA 需要更新 scaler 状态



'''
让 MPS 跑得更快的 3 个实用小技巧（基于你的提速需求）

既然你已经感受到了提速，再补充几个细节，能让训练 / 推理速度再上一个台阶：
1. 开启「通道最后（channels_last）」内存格式

MPS 对 NHWC（通道最后）格式的张量运算优化更好，默认的 NCHW 格式会有额外的格式转换开销：
'''
# 初始化模型后，转换为channels_last格式
model = nn.Linear(100, 200).to("mps")
model = model.to(memory_format=torch.channels_last)

# 输入张量也同步转换格式
x = torch.randn(32, 100).to("mps").to(memory_format=torch.channels_last)
output = model(x)

'''
2. 避免「小张量频繁在 CPU/MPS 间切换」

MPS 和 CPU 之间的数据传输（to("cpu")/to("mps")）有额外开销，尤其是小批次、小张量的频繁切换，会抵消 MPS 的加速效果：
'''
# ❌ 不好的写法：每次循环都切换设备
for x in dataloader:
    x = x.to("mps")  # 频繁切换
    output = model(x)
    loss = loss_fn(output, y.to("mps"))
    loss.backward()
    print(loss.item())  # 隐式把loss从mps转到cpu

# ✅ 好的写法：批量转移+减少切换
# 1. 数据加载时直接加载到MPS（如果内存够）
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)
# 2. 批量处理完再统一转回CPU打印/保存
losses = []
for x, y in dataloader:
    x, y = x.to("mps"), y.to("mps")
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    losses.append(loss.detach())  # 先存在MPS上
# 3. 最后统一转回CPU
losses = torch.tensor(losses).to("cpu")
print(f"平均损失: {losses.mean().item()}")

'''
3.调整批次大小（batch size）到「MPS 最优值」

M 芯片的 GPU 核心数是固定的（比如 M1 有 8 核 GPU，M2 有 10 核），批次太小会导致 GPU 核心利用率不足，太大又会触发内存交换：
建议测试 batch size = 16/32/64/128，找到「速度最快 + 不爆内存」的平衡点；
比如你的模型 batch_size=32 时速度最快，就固定用这个值，比默认的 8/16 能提升 20%-30%。
'''

'''
补充：MPS 目前的小局限（避免踩坑）

虽然提速明显，但 MPS 还不是「完美替代 CUDA」，比如：
部分算子不支持（比如torch.nn.functional.gelu的某些变体、复杂的自定义算子），会自动回退到 CPU，导致速度波动；
大模型（比如 7B 以上的 LLM）训练时，MPS 的显存管理不如 CUDA 灵活，容易出现 OOM；
混合精度下，部分 FP16 算子的精度稳定性不如 CUDA，偶尔会出现 Loss 震荡。
不过对你的端侧场景来说，这些局限基本不影响 —— 端侧模型通常是轻量级的（比如 CNN 分类、小 Transformer），MPS 的优势完全能覆盖这些小问题。
总结

提速核心原因：MPS 把运算从 M 芯片的 CPU 转移到 GPU 核心（Metal 加速），而 TensorFlow 早期适配滞后，只能依赖 CPU 硬跑，两者的速度差本质是「GPU 并行运算 vs CPU 串行运算」的差距；
进一步提速技巧：开启 channels_last 格式、减少设备间数据切换、调整最优 batch size，能让 MPS 的速度再提升 10%-30%；
适用场景：MPS 对端侧轻量级模型的提速效果最明显，完全能满足日常开发需求。
'''