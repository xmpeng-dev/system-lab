"""
axion/feature0/code/comm_report.py

CommReport: 聚合 StepStats，计算统计指标，生成 HTML 可视化。

依赖:
    - numpy
    - plotly (可选，只用于 save_html)
"""

import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List


# ─────────────────────────────────────────────────────────────
# 单层报告
# ─────────────────────────────────────────────────────────────

@dataclass
class LayerReport:
    layer_idx: int
    # Expert 负载
    mean_tokens_per_expert: float
    max_tokens_per_expert: float
    load_imbalance_ratio: float        # max / mean，理想值 = 1.0
    hot_experts: List[int]             # 负载 top-5 的 expert id
    # A2A 时间（ms）
    dispatch_a2a_ms_mean: float
    combine_a2a_ms_mean: float
    a2a_total_ms_mean: float           # dispatch + combine
    moe_layer_ms_mean: float
    a2a_fraction: float                # a2a_total / moe_layer，核心决策指标
    # 带宽
    dispatch_gbps: float               # 实测 dispatch 带宽（GB/s）
    combine_gbps: float                # 实测 combine 带宽（GB/s）


# ─────────────────────────────────────────────────────────────
# 全局报告
# ─────────────────────────────────────────────────────────────

@dataclass
class CommReport:
    num_layers: int
    num_profiled_steps: int
    layers: List[LayerReport] = field(default_factory=list)

    # ── 全局汇总属性 ──────────────────────────────────────────

    @property
    def global_load_imbalance(self) -> float:
        """所有层的平均负载不均衡系数（决定 Feature 1/2 优先级）"""
        if not self.layers:
            return 1.0
        return float(np.mean([l.load_imbalance_ratio for l in self.layers]))

    @property
    def global_a2a_fraction(self) -> float:
        """所有层的平均 A2A 时间占比（决定 Feature 3/4 优先级）"""
        if not self.layers:
            return 0.0
        return float(np.mean([l.a2a_fraction for l in self.layers]))

    @property
    def mean_dispatch_gbps(self) -> float:
        """平均 dispatch 带宽（判断是否有大量跨节点流量）"""
        if not self.layers:
            return 0.0
        return float(np.mean([l.dispatch_gbps for l in self.layers]))

    # ── 构建方法 ──────────────────────────────────────────────

    @classmethod
    def from_stats(cls, stats: list) -> "CommReport":
        """从 StepStats 列表聚合生成 CommReport"""
        # 按 layer_idx 分组
        layer_data = defaultdict(list)
        for s in stats:
            layer_data[s.layer_idx].append(s)

        if not layer_data:
            return cls(num_layers=0, num_profiled_steps=0)

        num_layers = len(layer_data)
        num_steps  = max(len(v) for v in layer_data.values())

        layers = []
        for layer_idx in sorted(layer_data.keys()):
            step_list = layer_data[layer_idx]

            # ── Expert 负载统计（多步平均）
            valid_tpe = [s.tokens_per_expert.numpy()
                         for s in step_list
                         if s.tokens_per_expert.numel() > 1]
            if valid_tpe:
                all_tpe  = np.stack(valid_tpe)        # [steps, num_experts]
                mean_tpe = all_tpe.mean(axis=0)       # [num_experts]
                max_tpe  = float(mean_tpe.max())
                mean_val = float(mean_tpe.mean())
                imbalance = max_tpe / mean_val if mean_val > 0 else 1.0
                hot_experts = list(map(int, np.argsort(mean_tpe)[::-1][:5]))
            else:
                max_tpe = mean_val = 0.0
                imbalance = 1.0
                hot_experts = []

            # ── A2A 时间统计
            d_ms   = float(np.mean([s.dispatch_a2a_ms for s in step_list]))
            c_ms   = float(np.mean([s.combine_a2a_ms  for s in step_list]))
            moe_ms = float(np.mean([s.moe_layer_ms    for s in step_list]))
            a2a_ms = d_ms + c_ms
            a2a_frac = a2a_ms / moe_ms if moe_ms > 0 else 0.0

            # ── 带宽估算
            d_bytes = float(np.mean([s.dispatch_bytes for s in step_list]))
            c_bytes = float(np.mean([s.combine_bytes  for s in step_list]))
            d_gbps  = d_bytes / (d_ms * 1e-3) / 1e9 if d_ms > 0 else 0.0
            c_gbps  = c_bytes / (c_ms * 1e-3) / 1e9 if c_ms > 0 else 0.0

            layers.append(LayerReport(
                layer_idx              = layer_idx,
                mean_tokens_per_expert = mean_val,
                max_tokens_per_expert  = max_tpe,
                load_imbalance_ratio   = imbalance,
                hot_experts            = hot_experts,
                dispatch_a2a_ms_mean   = d_ms,
                combine_a2a_ms_mean    = c_ms,
                a2a_total_ms_mean      = a2a_ms,
                moe_layer_ms_mean      = moe_ms,
                a2a_fraction           = a2a_frac,
                dispatch_gbps          = d_gbps,
                combine_gbps           = c_gbps,
            ))

        report = cls(num_layers=num_layers, num_profiled_steps=num_steps, layers=layers)
        report._print_summary()
        return report

    # ── 输出方法 ──────────────────────────────────────────────

    def _print_summary(self):
        """rank 0 打印摘要到 stdout"""
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass

        W = 65
        print("\n" + "=" * W)
        print("  AxionCommProfiler Report")
        print("=" * W)
        print(f"  Layers profiled          : {self.num_layers}")
        print(f"  Steps profiled           : {self.num_profiled_steps}")
        print(f"  Global load imbalance    : {self.global_load_imbalance:.2f}x  "
              f"({'HIGH ⚠' if self.global_load_imbalance >= 2.0 else 'OK'})")
        print(f"  Global A2A fraction      : {self.global_a2a_fraction * 100:.1f}%  "
              f"({'HIGH ⚠' if self.global_a2a_fraction >= 0.2 else 'OK'})")
        print(f"  Mean dispatch bandwidth  : {self.mean_dispatch_gbps:.1f} GB/s")
        print("-" * W)
        print(f"  {'Layer':>5}  {'Imbal':>7}  {'A2A(ms)':>9}  {'A2A%':>6}  "
              f"{'Disp BW':>8}  {'HotExp'}")
        print("-" * W)
        for l in self.layers:
            print(f"  {l.layer_idx:>5}  "
                  f"{l.load_imbalance_ratio:>6.2f}x  "
                  f"{l.a2a_total_ms_mean:>8.1f}  "
                  f"{l.a2a_fraction * 100:>5.0f}%  "
                  f"{l.dispatch_gbps:>7.1f}  "
                  f"{l.hot_experts[:3]}")
        print("=" * W)
        # 决策门输出
        print("\n  ── Decision Gate ──")
        if self.global_load_imbalance >= 2.0:
            print("  [!] Load imbalance ≥ 2.0x → Feature 1 (FastRouter) HIGH PRIORITY")
        elif self.global_load_imbalance < 1.3:
            print("  [√] Load imbalance < 1.3x → Feature 1/2 LOW priority")
        if self.global_a2a_fraction >= 0.20:
            print("  [!] A2A fraction ≥ 20%   → Feature 3 (Overlap) HIGH PRIORITY")
        elif self.global_a2a_fraction < 0.10:
            print("  [√] A2A fraction < 10%   → Feature 3/4 LOW priority")
        print()

    def save_json(self, path: str):
        """保存为 JSON（供后续分析脚本读取）"""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[CommReport] JSON saved to {path}")

    def save_html(self, path: str):
        """
        生成 4 格可视化 HTML（基于 Plotly）：
          1. Expert 负载不均衡系数（逐层）
          2. A2A 时间占比（逐层）
          3. Dispatch vs Combine A2A 耗时（逐层）
          4. 实测 A2A 带宽（逐层）
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("[CommReport] plotly not installed. Run: pip install plotly")
            return

        layer_ids = [l.layer_idx for l in self.layers]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Expert Load Imbalance (max/mean per layer)",
                "A2A Time Fraction (% of MoE layer time)",
                "Dispatch vs Combine A2A Time (ms)",
                "Effective A2A Bandwidth (GB/s)",
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        # 1. 负载不均衡系数
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.load_imbalance_ratio for l in self.layers],
            name="Load Imbalance",
            marker_color="crimson",
            showlegend=True,
        ), row=1, col=1)
        # 参考线：理想值 = 1.0
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="ideal=1.0", row=1, col=1)

        # 2. A2A 占比
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.a2a_fraction * 100 for l in self.layers],
            name="A2A Fraction (%)",
            marker_color="royalblue",
        ), row=1, col=2)
        fig.add_hline(y=20.0, line_dash="dash", line_color="orange",
                      annotation_text="threshold=20%", row=1, col=2)

        # 3. Dispatch vs Combine
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.dispatch_a2a_ms_mean for l in self.layers],
            name="Dispatch A2A (ms)",
            marker_color="darkorange",
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.combine_a2a_ms_mean for l in self.layers],
            name="Combine A2A (ms)",
            marker_color="seagreen",
        ), row=2, col=1)

        # 4. 带宽
        fig.add_trace(go.Scatter(
            x=layer_ids,
            y=[l.dispatch_gbps for l in self.layers],
            name="Dispatch BW (GB/s)",
            mode="lines+markers",
            line=dict(color="darkorange"),
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=layer_ids,
            y=[l.combine_gbps for l in self.layers],
            name="Combine BW (GB/s)",
            mode="lines+markers",
            line=dict(color="seagreen"),
        ), row=2, col=2)

        fig.update_layout(
            title=dict(
                text=(f"AxionCommProfiler: MoE Communication Analysis<br>"
                      f"<sup>Layers={self.num_layers}, "
                      f"Steps={self.num_profiled_steps}, "
                      f"GlobalImbalance={self.global_load_imbalance:.2f}x, "
                      f"GlobalA2A={self.global_a2a_fraction*100:.1f}%</sup>"),
                x=0.5,
            ),
            height=800,
            barmode="group",
        )

        fig.write_html(path)
        print(f"[CommReport] HTML report saved to {path}")
