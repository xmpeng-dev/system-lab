"""
axion/feature0/code/example_megatron.py

在 Megatron-LM pretrain 脚本中接入 AxionCommProfiler 的示例。
只需修改 3 处（标记 ← ADD）。
"""

import torch
import torch.distributed as dist

# ── 原有 Megatron 导入 ────────────────────────────────────────
# from megatron.training import pretrain
# from megatron.core.models.gpt import GPTModel
# ... 省略其余 Megatron 导入

# ← ADD 1：导入 profiler
from comm_profiler import AxionCommProfiler


def model_provider(pre_process=True, post_process=True):
    """原有的 model_provider，不需要修改"""
    # model = GPTModel(...)
    # return model
    pass


def forward_step(data_iterator, model):
    """原有的 forward_step，不需要修改"""
    pass


def train(model, optimizer, lr_scheduler, train_data_iterator, ...):
    """
    原有训练循环示例——只需加 3 处。
    """
    # ← ADD 2：在 model 构建完成后，创建并挂载 profiler
    profiler = AxionCommProfiler(
        num_warmup_steps=5,    # 前 5 步 warmup，不采样
        profile_steps=20,      # 只采样 20 步，不影响长期训练
        enabled=True,
    )
    profiler.attach(model)

    for step in range(total_steps):
        # 正常的训练步骤（不需要改）
        loss = forward_step(data_iterator, model)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # ← ADD 3（可选）：在采样完成后立即生成报告并停止
        if profiler.num_collected_steps >= profiler.profile_steps:
            report = profiler.report()
            if dist.get_rank() == 0:
                report.save_html("mi300x_moe_comm_report.html")
                report.save_json("mi300x_moe_comm_report.json")
            profiler.detach()   # 移除所有 hook，恢复模型原始状态
            break               # 或者继续训练（已经 detach，无额外开销）

    # 也可以在训练结束后再统一生成报告：
    # report = profiler.report()
    # ...


# ── 最简接入示例（伪代码）─────────────────────────────────────

def minimal_example():
    """
    如果只想快速验证 profiler 能否正常挂载，
    用这个最小示例（不依赖完整 Megatron 训练脚本）。
    """
    # 1. 假设已有 Megatron model
    # model = build_model(...)

    # 2. 创建 profiler
    profiler = AxionCommProfiler(num_warmup_steps=2, profile_steps=5)
    profiler.attach(model)

    # 3. 跑几步训练
    for _ in range(10):
        outputs = model(dummy_input)
        outputs.sum().backward()

    # 4. 生成报告
    report = profiler.report()
    report.save_html("/tmp/test_report.html")
    report.save_json("/tmp/test_report.json")

    profiler.detach()
    print("Done. Check /tmp/test_report.html")
