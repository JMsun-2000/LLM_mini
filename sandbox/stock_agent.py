import time
from datetime import datetime

# ===================== 你只需要改这里 =====================
def call_ai(prompt: str) -> str:
    # 替换成你的 API 调用
    response = "模型返回内容"
    return response

def send_notify(content: str):
    # 替换成你的推送（企业微信/钉钉/Server酱/邮箱）
    print("\n" + "="*50)
    print(content)
    print("="*50)
# ==========================================================

# ===================== Agent 固定 Prompt =====================
PROMPT_MOUTAI = """
你是专业股票分析Agent，只输出精简、可直接操作的结论。
现在按以下5点实时分析贵州茅台：
1. 北上资金动向
2. 飞天茅台批价状态
3. 成交量、换手率（判断真假涨跌）
4. 白酒板块与大盘环境
5. 关键支撑位、压力位

输出严格按以下格式，不要多余内容：
【今日状态】
【核心原因】
【北上资金】
【批价】
【量能】
【真假涨跌】
【支撑位】
【压力位】
【操作建议】
【一句话结论】
""".strip()

PROMPT_BAIJIU = """
你是专业股票分析Agent，只输出精简、可直接操作的结论。
实时分析白酒板块（白酒指数）：
1. 指数当前强弱
2. 成交量状态
3. 领涨/领跌逻辑
4. 对茅台影响

输出严格按以下格式：
【白酒板块状态】
【量能】
【板块强弱】
【对茅台影响】
【结论】
""".strip()

PROMPT_DAPAN = """
你是专业股票分析Agent，只输出精简、可直接操作的结论。
实时分析大盘（上证50、沪深300、权重情绪）：
1. 大盘强弱
2. 权重是否护盘
3. 市场整体风险

输出严格按以下格式：
【大盘状态】
【权重情绪】
【市场风险】
【对茅台影响】
【结论】
""".strip()

# ===================== 风险判断逻辑 =====================
def check_risk(moutai_text: str) -> str:
    """自动识别风险：破位、真跌、放量出逃 → 标红警告"""
    warning = []
    text = moutai_text.lower()

    if "真跌" in text:
        warning.append("🚨 真跌，注意风险")
    if "破位" in text:
        warning.append("🚨 破位，警惕跳水")
    if "放量下跌" in text:
        warning.append("🚨 放量下跌，资金出逃")
    if "减仓" in text:
        warning.append("⚠️ 建议减仓")

    if warning:
        return "【⚠️ 紧急预警】" + " | ".join(warning)
    return "【✅ 正常】"

# ===================== 只在交易时间运行 =====================
def is_trading_time():
    now = datetime.now()
    h, m = now.hour, now.minute
    # 9:30 ~ 11:30, 13:00 ~ 15:00
    if (9 < h < 11) or (11 == h and m <= 30):
        return True
    if (13 <= h < 15):
        return True
    return False

# ===================== 主巡检 =====================
def check_all():
    if not is_trading_time():
        return
        
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== {now} 开始巡检 ===")

    moutai = call_ai(PROMPT_MOUTAI)
    baijiu = call_ai(PROMPT_BAIJIU)
    dapan  = call_ai(PROMPT_DAPAN)

    # 风险预警
    risk = check_risk(moutai)

    msg = (
        f"{risk}  {now}\n\n"
        "🍶 茅台巡检\n" + moutai + "\n\n"
        "🥂 白酒板块\n" + baijiu + "\n\n"
        "📊 大盘环境\n" + dapan
    )

    send_notify(msg)

# ===================== 启动 =====================
if __name__ == "__main__":
    print("🚀 终极AI盯盘已启动 | 10分钟一轮 | 自动风险预警")
    while True:
        check_all()
        print("\n等待10分钟...")
        time.sleep(600)