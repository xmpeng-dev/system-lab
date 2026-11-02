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
| 2026-10-28 | [`2026-10-28_2150_megamoe-paper-skeleton.md`](./2026-10-28_2150_megamoe-paper-skeleton.md) | **【论文骨架本体 —— 直接往里写】**Title / Abstract(草稿,数字已核实可直接用)/ §1–§10 逐节逐段的写作提示与出处 / 附录 A–D / 删除清单 / TODO 汇总表。空缺一律留 `TODO` 不编造。关键点:**§3 "The Deployed System" 整节空白** —— 我已核实的部分(`use_turbo_mega_moe` **默认关闭**、`turbo_mega_moe_precision` bf16\|mxfp8、配置注释限定 EP-only/TP=1、接入点在 `backends/megatron/core/extensions/mega_moe.py` 且**调用的就是 stage1/stage2 两阶段拆分**)已写入,其余列成 **T3-1~T3-7 七问**需你提供;**鉴于默认关闭,资格类别上"roadmap 上的在建系统"大概率比"已部署"更诚实,而工业轨同样接受**。另:§7.4 发现 blog 引的两个 `rocm.blogs.amd.com` 支撑结果其一未发布,论文不能引不存在的文献,需重测或删除 |
| 2026-10-28 | [`2026-10-28_2140_megamoe-paper-outline.md`](./2026-10-28_2140_megamoe-paper-outline.md) | **【页数预算与图表取舍】双栏 10 页的逐节框架** —— 每节标页数预算、已有材料的 blog 出处、必须新增的内容;附 §7 Evaluation 七个小节与 §8 六条教训的细分、图表取舍方案、"补什么"的优先级排序。搭框架时暴露的三个新问题:①**blog 的不均衡路由稳健性论据引自 `primus-moe-optimization` 这篇未发布文章**(连同两个 `rocm.blogs.amd.com` 死链)⇒ 该组实验必须**搬进论文自己做**,优先级从"可选"升为"硬补" ②**L5(`#486` vendoring 教训)blog 里根本没有**,证据在 `flydsl/mega/bf16/__init__.py:14-24` 注释里 ⇒ **零实验、纯写作,全篇性价比最高** ③10 页装不下 blog 的 12 张表,方案=5 图全留 + 主体 5 表(含新的"实现机制合并表"),低精度五个子节压成一段,但**因附录审稿人无义务读,核心结论不得下放**。另:§1 贡献列表必须是「教训+方法学+系统」而非加速比,且前向引用 §8 |
| 2026-10-28 | [`2026-10-28_2115_megamoe-paper-mlsys-plan.md`](./2026-10-28_2115_megamoe-paper-mlsys-plan.md) | **【路线与规则】路线 A = MLSys 工业轨生产系统论文**(用户已定)。①**工业轨只隐作者名、公司名/产品名/URL 全可保留** ⇒ AMD / MI355X / Primus / MegaMoE 照写,blog 内容近乎逐字迁移 ②**平台特异性从劣势翻成资产** —— gfx950 的 visibility scope、XCD 感知 CU 划分正是工业轨要的"真实平台经验" ③**新硬门槛**:必须描述真实系统(已部署回顾 / roadmap 上在建 / 带教训的已取消),且第一作者与多数作者须来自业界 ⇒ **需新写"部署刻画"一节,blog 完全没有** ④评审标准是 impact/lessons/experiences 不是 novelty ⇒ 差异化在**六条教训**(L1 框架设定融合上界 / L2 可证伪的重叠测量口径 / L3 精度测试挡不住一致性 bug / L4 融合不总省显存 / **L5 静默数值回归的答案是 vendoring —— 研究轨不会接受、工业轨明确欢迎** / L6 liveness 是假设非证明)⑤**截止后不可换轨**;双栏 10 页;附录审稿人无义务读;作者配额待查 ⑥约 6 个月工作量(A1 收敛=T1-4、A2 配置补齐、A3 部署刻画、A4 同硬件对照=T3-3、A5 教训成文),余 5 个月抬高度(A6 gfx1250 第二平台对冲过时、A7 liveness 压力实验)⑦**更正**:AE 大概率是录取后自愿环节、不影响决定,不能当过线手段 ⑧抢发风险大降、过时与审批风险上升 |
| 2026-10-28 | [`2026-10-28_2050_megamoe-paper-plan.md`](./2026-10-28_2050_megamoe-paper-plan.md) | **MegaMoE 论文立项**(roadmap T3-3 的展开)。⓪**blog 未发布**(只在内部仓分支 `xiaompen_amdeng-primus-mega-moe` 上,不在 `origin/develop`;frontmatter 的 `15 Sep 2026` 是草稿计划日期)⇒ 双盲不是障碍、自引也不构成在先技术(`primus-moe-optimization` 同样未发布,唯一已并入 develop 的是 `primus-deepseek-v4`);但**新增两个卡点**:blog 上线与投稿的排序冲突、以及同一批数据的对外发表审批 ①**窗口只有两个活的**:MLSys'27 = **2026-10-30 20:00 UTC**(后天),OSDI'27 摘要 = 12-01;EuroSys'27 / ASPLOS'27 / PPoPP'27 全部 cycle 已关 ②**MLSys'27 工业轨原文明写 "need NOT present new ideas or novel solutions"** —— 现有 blog 材料(561 行,约 85% 可直接复用)唯一能两天内合法投出去的地方 ③**这是两篇论文不是一篇**:Paper A = 工业轨,不声称机制新颖性;Paper B = 以「**融合粒度的上界由上层框架的重叠机制决定**」为 thesis 的研究论文(blog 把这条埋在 Engineering Trade-Offs 里,但整个 megakernel 文献无人讨论过这个天花板,且有"算子变快、step 不动"的实测反例)④**Triton-distributed 对照跑得了,但也废掉一个自以为的新颖点**:本地 checkout 有 `kernels/amd/ep_all2all_fused.py`(1,267 行)+ `test/amd/`,`function/amd/ep_moe_fused.py:80-82` 是 `raise NotImplementedError` —— **只有前向**,于是 blog delta 第 5 条(前反向共享通信原语)升格为唯一可比项;**但它的前向切分 docstring 与 MegaMoE 的两算子切分一字不差且已在 AMD 开源**,该切分不能当贡献写。附 E1–E5 实验方案与三个不可比点 ⑤blog 的**测量方法学**(双基线 + η + 用等效 FLOP 稠密 GEMM 反证分母 + 抓到精度全绿时 bug 的确定性哈希)可当第三贡献,成本近零 |
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
