# primus-turbo — Primus-Turbo 库本身的 owner 视角笔记

> Primus-Turbo 作为**一个被交付的算子库**的规划与审计记录。
> 算子/kernel 的逐次优化记录在兄弟目录:`slab/notes/MegaMoeFlydsl/`、`slab/notes/hk-attn-bwd/`、`slab/notes/peer-tiles/`、`slab/notes/moe_perf/`。
> 把 Turbo 当作 Primus MoE 故事的一个组件来看时,去 `slab/notes/primus-moe/`。

## Status

| 维度 | 值 |
|---|---|
| Owner | Xiaoming Peng(2026-10 接任) |
| Repo | `~/workspace/Primus-Turbo/` — `AMD-AGI/Primus-Turbo`,默认分支 `main` |
| 唯一消费者 | `~/workspace/Primus/`(66 行顶层 `from primus_turbo` import,24 行带守卫) |
| Hardware | gfx942(MI300X/MI325X)、gfx950(MI350X/MI355X)、**gfx1250(MI455X / Helios)** |
| Kernel 栈 | FlyDSL(主力,85K 行)、Triton(主力,15K 行)、csrc HIP/CK(55K 行)、Gluon / JAX / Python-HipKittens(均休眠) |
| CI 覆盖 | 只构建 `gfx942;gfx950`;**gfx1250 默认全跳**,12 条 allow-list 放行;PR 上 GPU job 需 `ci:gpu` label |
| 性能门 | **无**(nightly 为 `continue-on-error`) |
| Last roadmap | [2026-10-28 接任 owner 版 2026Q4–2027H1](./2026-10-28_primus-turbo-roadmap-2026q4-2027h1.md) |

## Index

| 日期 | 文件 | 内容 |
|---|---|---|
| 2026-10-28 | [`2026-10-28_primus-turbo-roadmap-2026q4-2027h1.md`](./2026-10-28_primus-turbo-roadmap-2026q4-2027h1.md) | **接任 owner 的首版 roadmap** — 五路只读 subagent 全仓审计 + 手工复核,结合 [2026-10-28 业界扫描](../primus-moe/2026-10-28_0546_industry-scan-update.md)。核心判断:①**最严重的问题不在 kernel 在契约** —— Primus 有 ≥3 个 import 指向在 Turbo **任何分支上都不存在**的模块(ODC primitives 是裸顶层 import,直接 ImportError),Turbo 无版本约束,且 Primus 深入到 `kernels/*_impl`、`core.backend`、`_C` 三个私有层 ②**CI 不测团队正在做的硬件** —— 主力分支 59 个 commit 全是 gfx1250,而 CI 只建 gfx942+gfx950、gfx1250 默认全跳;上次 release validate 是红的(29 个 attention SNR 失败)③**对静默数值回归的现行答案是 vendoring 不是测试** —— `#486` 重建共享 FlyDSL BF16 grouped GEMM 静默把 MegaMoE 的 it1 grad norm 从 1.45 打到 1.88e6,处置是把 tile 钉快照私有化,现已付出三套 mega 栈 / 三套 symm_buffer 的代价;**这是「前向对、无 NaN、dW 错」失效家族的第四例** ④两个真实资产被低估:MegaMoE 的融合深度(fp8 pool 直接 in-register colwise requant 不走 bf16 往返,公开实现里没见过)与 MXFP4(dist_moe 明写不支持,是唯一能说「我有你没有」的格子),但格式缝成果**困在 MoE 路径里**,通用 `gemm_fp8/fp4` 全是独立 launch ⑤无任何性能门。分期 **T0 契约止血 → T1 可信度 → T2 收敛六栈 → T3 差异化 → T4 通信层决断**;唯一硬依赖是 **T2-5(de-vendor mega)必须在 T1-4(梯度范数回归门)之后**;若只做三件事 = T1-4 + T0-2 + T1-1 |

## Cross-references

- **业界基线**:[`../primus-moe/2026-10-28_0546_industry-scan-update.md`](../primus-moe/2026-10-28_0546_industry-scan-update.md)
- **Primus 框架侧 roadmap**:[`../primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md`](../primus-moe/2026-09-03_primus-roadmap-2026q4-2027h2.md) + [10-12 delta](../primus-moe/2026-10-12_0101_primus-roadmap-update.md)
- **格式缝(职业主线)**:[`../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md`](../career-strategy/2026-08-12_1505_the-quant-gemm-seam.md)
- **MegaMoE FlyDSL 优化记录**:`../MegaMoeFlydsl/`
- **活跃 bug(T1-4 的直接论据)**:`/perf_apps/xiaoming/Primus/HANDOFF_mxfp8_topk_grad.md`
- **kernel 优化迭代硬约束**:`Primus-Turbo/.cursor/rules/iteration_rules.mdc`

## Maintenance

- 一个 release cycle 一份 roadmap;旧的原地保留,最新的从 `Status` 行链出。
- 新 roadmap 落地时同步更新 `Status` 表与 `Index` 表。
- 本目录只放**库作为交付物**的规划/审计;单个 kernel 的优化轮次记录去对应的 kernel 目录。
