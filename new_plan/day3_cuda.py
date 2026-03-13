import torch
import torch.cuda.amp as amp  # 混合精度核心库

# ====================== 一、CUDA 基础检查（先确认GPU环境）======================
print("===== 1. CUDA 基础检查 =====")
# 1. 检查是否有可用的CUDA设备（GPT训练/端侧推理都需要先判断）
has_cuda = torch.cuda.is_available()
print("是否有CUDA设备:", has_cuda)

    # 替代CUDA的设备判断（Mac专属）
def get_universal_device():
    """
    全平台适配的设备选择函数（字节端侧LLM代码标准写法）
    优先级：CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")



# 2. 查看CUDA设备数量和名称（多GPU场景会用到）
if has_cuda:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print(f"CUDA设备数量: {device_count}")
    print(f"当前使用的GPU: {device_name}")


# 3. 定义设备（核心！所有张量/模型都要绑定设备）
# 优先用GPU，没有就用CPU（端侧训练时手机一般用CPU/NPU，逻辑一致）
device = torch.device("cuda:0" if has_cuda else "cpu")

# 测试：替换Day3的device定义
# 全局设备变量，后续所有代码都用这个
device = get_universal_device()
print(f"最终使用的设备: {device}")

# ====================== 二、张量的设备迁移（GPT核心操作）======================
print("\n===== 2. 张量的设备迁移 =====")
# 1. 创建CPU张量（模拟GPT的初始输入张量）
cpu_tensor = torch.randn(2, 3, 4)  # [batch, seq_len, hidden_dim]
print("CPU张量的设备:", cpu_tensor.device)

# 2. 迁移到GPU（两种等价写法，GPT代码里常用第一种）
gpu_tensor = cpu_tensor.to(device)
# gpu_tensor = cpu_tensor.cuda()  # 第二种写法（仅CUDA可用时）
print("迁移到GPU后的设备:", gpu_tensor.device)

# 3. 直接在GPU上创建张量（更高效，避免CPU→GPU拷贝）
direct_gpu_tensor = torch.randn(2, 3, 4, device=device)
print("直接创建的GPU张量设备:", direct_gpu_tensor.device)

# 4. 张量运算的设备一致性（GPT里最容易踩的坑）
# 错误示例：CPU张量和GPU张量不能运算（注释掉，运行会报错）
# wrong_tensor = cpu_tensor + gpu_tensor

# 正确做法：所有张量统一设备
cpu_tensor2 = cpu_tensor.to(device)
correct_tensor = gpu_tensor + cpu_tensor2
print("同设备张量运算结果形状:", correct_tensor.shape)

# 5. 从GPU迁回CPU（端侧训练后同步到云端时会用到）
back_to_cpu = gpu_tensor.cpu()
print("迁回CPU后的设备:", back_to_cpu.device)

# ====================== 三、模型的设备迁移（GPT模型部署核心）======================
print("\n===== 3. 模型的设备迁移 =====")
# 定义一个极简的GPT小模型（模拟后续要写的Transformer）
class TinyGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)  # 模拟隐藏层
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# 1. 创建模型（默认在CPU）
model = TinyGPT()
print("模型初始设备:", next(model.parameters()).device)

# 2. 迁移模型到GPU（核心！训练/推理前必须做）
model.to(device)
print("模型迁移后的设备:", next(model.parameters()).device)

# 3. 模型推理（输入张量必须和模型同设备）
input_tensor = torch.randn(2, 3, 4, device=device)
output = model(input_tensor)
print("模型输出形状:", output.shape)
print("模型输出设备:", output.device)

# ====================== 四、混合精度训练（GPT提速降显存的关键）======================
print("\n===== 4. 混合精度训练（AMP） =====")
# 混合精度核心：用FP16（半精度）做前向，FP32（单精度）做反向，兼顾速度和精度
# 字节的LLM训练/端侧推理都用这个优化，能省50%显存，提速30%+

# 1. 初始化混合精度上下文（GradScaler是核心，防止梯度下溢）
scaler = amp.GradScaler() if has_cuda else None

# 2. 模拟GPT训练循环（混合精度版）
if has_cuda and scaler is not None:
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 混合精度训练步骤（GPT训练的标准流程）
    for step in range(2):  # 模拟2步训练
        optimizer.zero_grad()  # 清空梯度
        
        # 前向传播（用autocast开启混合精度）
        with amp.autocast(device_type="cuda"):
            pred = model(input_tensor)
            loss = torch.mean(pred)  # 模拟损失函数
        
        # 反向传播（用scaler缩放梯度，避免FP16梯度下溢）
        scaler.scale(loss).backward()
        
        # 更新参数（先反缩放梯度，再优化）
        scaler.step(optimizer)
        
        # 更新scaler（自适应调整缩放系数）
        scaler.update()
        
        print(f"Step {step+1} | Loss: {loss.item():.4f} | 混合精度训练完成")
else:
    print("无CUDA设备，跳过混合精度演示（CPU下混合精度收益低）")

# ====================== 五、端侧适配小技巧（贴合你的端云协同方案）======================
print("\n===== 5. 端侧CUDA/NPU适配技巧 =====")
# 1. 设备无关代码（手机端可能用CPU/NPU，代码要兼容）
def get_device():
    """适配端侧的设备选择函数（字节端侧LLM代码标准写法）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # 可扩展：适配手机NPU（如骁龙AI、苹果Neural Engine）
    # elif has_npu():
    #     return torch.device("npu")
    else:
        return torch.device("cpu")

# 2. 轻量化设备迁移（端侧Adapter训练用）
def move_to_device(data, device):
    """统一迁移张量/模型到指定设备（端侧代码复用性高）"""
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)  # non_blocking加速端侧拷贝
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    else:
        return data

# 测试设备无关代码
test_device = get_device()
test_tensor = torch.randn(1, 1, 4)
test_tensor = move_to_device(test_tensor, test_device)
print(f"端侧适配后张量设备: {test_tensor.device}")

# ====================== 六、Day3 小练习（验证掌握程度）======================
print("\n===== 6. Day3 小练习 =====")
# 练习目标：实现GPT张量的设备迁移+混合精度前向
# 1. 创建[batch=2, seq_len=5, hidden_dim=16]的张量
practice_tensor = torch.randn(2, 5, 16)
# 2. 迁移到可用设备
practice_tensor = practice_tensor.to(device)
# 3. 定义一个线性层（模拟GPT的Embedding层）
linear = torch.nn.Linear(16, 32).to(device)
# 4. 用混合精度做前向（如果有GPU）
if has_cuda:
    with amp.autocast(device_type="cuda"):
        practice_output = linear(practice_tensor)
    print(f"混合精度前向输出形状: {practice_output.shape}")
    print(f"输出数据类型（FP16）: {practice_output.dtype}")  # 应为torch.float16
else:
    practice_output = linear(practice_tensor)
    print(f"CPU前向输出形状: {practice_output.shape}")