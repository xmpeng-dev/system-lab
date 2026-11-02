# MegaMoE 论文框架(路线 A / MLSys 工业轨,双栏 10 页)

> **When**: 2026-10-28 21:40 UTC+8
> **决定**: 路线 A 已定。本份是**可直接往里写的骨架** —— 每节标注页数预算、已有材料的 blog 出处、必须新增的内容。
> **规则依据**: MLSys 工业轨细则见 [`2026-10-28_2115_megamoe-paper-mlsys-plan.md`](./2026-10-28_2115_megamoe-paper-mlsys-plan.md) §1
> **源材料核对**: 本份的章节映射、图表编号、全部数字均于 2026-10-28 从 `primus-mega-moe/README.md` 逐行核对,非凭记忆

## 先说三个在搭框架时才暴露出来的问题

1. **blog 的一个关键稳健性论据,来源是一篇尚未发布的 blog。** 正文四处引用 "the companion post" 来支撑"不均衡路由下层级比值变化 <1%",那篇是 `primus-moe-optimization` —— **未并入 `develop`,不存在**。同理,末尾引用的两个 `rocm.blogs.amd.com` URL(1.081×/1.138× 与四节点 12.0%)其中第一个指向同一篇未发布文章。**论文不能引用不存在的文献**,所以不均衡路由这一组实验必须**搬进论文自己做**,从"可选补强"升级为"硬补"。
2. **`#486` 的 vendoring 教训(L5)在 blog 里根本没有。** 它的证据在 Turbo 仓里(`primus_turbo/flydsl/mega/bf16/__init__.py:14-24` 的注释原文)。这意味着 L5 是**纯写作成本、零实验成本**,是全篇性价比最高的一节。
3. **10 页装不下 blog 的 12 张表。** 必须做合并与下放,方案见 §图表预算。注意工业轨规则里"附录审稿人没有义务读",所以**凡是支撑核心结论的东西都不能下放**。

---

## 章节骨架与页数预算

总计 10.25 页,超出 0.25 页,**优先从 §5 砍**(见该节备注)。

| § | 标题 | 页 | 已有材料(blog 出处) | 必须新增 |
|---|---|---|---|---|
| — | Abstract | 0.25 | 数字齐备 | 按 §1 的贡献口径重写 |
| **1** | Introduction | 1.0 | Background 开头 + `## Summary` | **贡献列表必须是「教训 + 方法学 + 系统」,不是加速比** —— 见下方注 |
| **2** | Background: Why the Routed-Expert Path Has No Slack | 0.75 | `### The Routed-Expert Forward Path Has No Slack`(A2A 占串行 46% = 11.31/24.58 ms;前向 dispatch 暴露 41% = 2.265/5.475 ms)+ `### Low Precision Does Not Make This Go Away` + MegaScale-MoE 的 43.6%/32% 作为跨厂商佐证 | 无 |
| **3** | **The Deployed System** | 0.5 | **无** | **全新。**工业轨资格条件所在:多少模型、跑了多久、集群规模、默认路径还是可选、谁在用。写不出来不符合 "real system" 资格 |
| **4** | Design: Two Fused Operators | 1.25 | `## Two Fused Operators` + 算子表 + `### Backward Is the Conjugate of Forward` + **Figure 1**(`mega-moe-conjugate-pipeline.png`,共轭前反向,FC2 变 K 的 wgrad 是唯一留在融合对之外的 GEMM) | 无 |
| **5** | Implementation on CDNA4 | 1.25 | `## How the Overlap Happens` 全部五个小节 + **Figure 2**(`mega-moe-block-roles-signals.png`)+ 三张 signal 表 + cache modifier 表(18 = `sc1\|nt` agent 域 / 19 = `sc0\|sc1\|nt` system 域)+ CU 划分表(`num_dispatch_cu` 16/32/16、`num_combine_cu` 64/16)+ `### Low Precision Inside a Fused Operator` 五个子节 | **低精度那五个子节必须压缩成一段**,这是超出的 0.25 页该砍的地方。量化点、XGMI 上传什么、scale 随 token 到达、combine split 随载荷移动、什么留高精度 —— 保留结论,细节下放附录 |
| **6** | Measurement Methodology | 0.75 | `### Setup`(MI355X gfx950,8 GPU 单节点,EP8/TP=1,XGMI,BF16 + MXFP8 E4M3 block32 E8M0,8192 token/rank)+ η 与 speedup 的定义 + 双基线 + `#### How Fast Is the GEMM Leg Itself` 的等效 FLOP 稠密 GEMM 反证 + **Figure 4** | **把它写成一节独立贡献**(L2)。blog 里这些散在 Experiments 各处,论文里要集中成"怎么让重叠声明可证伪" |
| **7** | Evaluation | 2.25 | 见下方 §7 细分 | 四组新实验,见下方 |
| **8** | **Experience and Lessons** | 1.5 | L1/L3/L4/L6 有材料,L2 已上移到 §6,**L5 完全没有** | **全篇差异化所在。**见下方 §8 细分 |
| **9** | Related Work | 0.5 | 仅 `## Acknowledgments` 里的溯源段 | **必须正式重写**,见下方注 |
| **10** | Conclusion | 0.25 | `## Summary` | — |
| — | AI Usage Disclosure | — | — | 2027 届起要求置于参考文献之后(**待核实**) |

### 注:§1 的贡献列表怎么写

工业轨不按 novelty 评分,**所以贡献列表不能是"我们快了 1.5 倍"**。应当是:

1. 一个部署在生产训练栈里的融合式 EP MoE 算子,在 CDNA4 上的完整设计与实现报告(§3–§5);
2. 一套让通算重叠声明可证伪的测量口径(§6)—— 对应工业轨评审标准里的 "influence on the MLSys community";
3. 六条来自真实构建过程的教训(§8),其中若干是负面结果。

加速比放在 abstract 和 §7,作为支撑证据而非贡献本身。**并且 §1 要前向引用 §8 的教训清单**,让审稿人在第一页就看到这篇的落点。

### 注:§9 Related Work 必须覆盖的

blog 只有致谢式溯源,论文要正面处理。必须写进去的两类:

- **同代工作与溯源**:MegaScale-MoE(EuroSys'26,blog 自陈 overlap structure 跟随它)、DeepGEMM(`barrier.py` 改编自 `barrier.cuh`、`symm_buffer.py` 改编自 `sym_buffer.cuh`/`mega_moe.cuh`)、DeepEP、Triton-distributed、`meta-pytorch/dist_moe`、MegaFlux(arXiv 2610.00671)。
- **MLSys 2026 同场前作,必须逐篇定位**:ParallelKittens(多 GPU kernel 通算重叠的通用原则)、FarSkip-Collective(NVIDIA,MoE A2A 重叠)、FP8-Flow-MoE(casting-free FP8 MoE 数据流 —— **与本文的量化融进 epilogue 最接近,重叠程度必须精读确认**)、MoEBlaze(MoE 训练内存墙,与 §8 L4 相邻)、TokenWeave、HipKittens、Wave(AMD 自己的 DSL)。

**"为什么是 FlyDSL 不是 Wave"要在这一节或 §5 正面回答。**

---

## §7 Evaluation 细分(2.25 页)

| 小节 | 页 | 材料 | 状态 |
|---|---|---|---|
| 7.1 正确性论证的验证 | 0.25 | `### Validating a Manual Correctness Argument`(确定性哈希,抓到 wait 常量 bug) | **已有**。结论留此处,教训写进 §8 L3 |
| 7.2 BF16 算子级重叠 | 0.5 | 5 stage 表(DSv3:1.42–1.54×、η 85.6–88.5%)+ **Figure 3**;扩到 V4-Pro/V4-Flash 共 15 stage 得 1.29–1.61×、η 82.7–94.2%,XGMI 359–373 GB/s = 理论单向出口 537.6 GB/s 的 67–69% | **已有** |
| 7.3 两个指标必须合看 | 0.25 | V4-Pro dispatch dgrad 1.716 ms 通信配 2.075 ms 计算 → 1.61×;同算子 wgrad 1.663 ms 藏进 4.492 ms → η 94.2% 但只有 1.29× | **已有**,极好的论据,别压缩 |
| 7.4 训练步 | 0.5 | 6 模型表(DSv3 / V4-Flash / GLM-5 / Kimi-K2 / MiniMax-M2.5 / Qwen3-235B-A22B),BF16 1.162–1.286×、MXFP8 1.169–1.350×;每模型跑 5 个 arm,表里只报 2 个同精度配方的配对 | **已有**。但必须保留 blog 的全部限定语:深度是缩减代理(4/60、8/43、6/78、4/60、2/62、8/94)、MBS/GBS 逐模型不同故**跨行不可比**、20 迭代、mock 输入 |
| **7.5 收敛性** | **0.25** | **无** | **硬补(A1)。**BF16 与 MXFP8 各一条曲线,对 DeepEP 路径做 loss parity。blog 自陈 "These short runs measure throughput, not convergence" —— 一篇声称部署了训练系统的论文不能停在这 |
| **7.6 不利配置** | **0.25** | **无** | **硬补(A2)。**① 开 shared experts 的配对(现在关掉它等于替非融合调度卸掉一条本可摆在 dispatch 旁边的计算腿)② **不均衡路由的步级,必须在本文内做** —— 现有的 <1% 论据引自未发布的 companion post,不能引用 |
| **7.7 同硬件对照** | **0.25** | **无** | **应做(A4)。**① Triton-distributed on MI355X:**仅前向**(`function/amd/ep_moe_fused.py:80-82` 是 `raise NotImplementedError`),且其前向切分 docstring 与本文两算子切分一字不差 —— 这个事实本身要写进来 ② DeepEP 的**算子级**计时(blog 只有层级/步级) |

---

## §8 Experience and Lessons 细分(1.5 页)

次序按"对读者的可迁移性"排,不按时间。

| # | 教训 | 页 | 材料来源 | 成本 |
|---|---|---|---|---|
| **L1** | **框架设定融合上界。** 整层包成单个 autograd Function 后算子更快、训练步没更快 —— Megatron 分布式优化器的 hook 失去挂载点,参数 all-gather 与梯度 reduce-scatter 退化成串行。处置是 `fused_mega_moe_stage1/stage2` 两阶段拆分 | 0.4 | `### Keeping DDP Overlap Alive in a Real Training Framework` + **Figure 5**(`mega-moe-ddp-overlap-trace.png`)。**该图注原文"No kernel changed between the two traces"是全篇最有力的一句**,要在正文里引用 | 零(已有) |
| **L5** | **静默数值回归的现行答案是 vendoring,不是测试。** `#486` 重建共享 FlyDSL BF16 grouped GEMM 围绕 `Mfma16x16x32` 与新 S2R loader,对 host 侧 grouped GEMM 无害,却静默把 MegaMoE BF16 的 it1 grad norm 从 ~1.45 打到 ~1.88e6;处置是把 tile 钉到 `f6d5ab68` 快照私有化 | 0.3 | **blog 没有。**证据在 `primus_turbo/flydsl/mega/bf16/__init__.py:14-24` 的注释原文 | **零实验、纯写作 —— 性价比最高的一节。**研究轨不会接受这种章节,工业轨 CFP 明写欢迎 |
| **L2'** | 指向 §6(方法学已上移),此处只留一句"为什么我们认为现有报告口径不够" | 0.1 | §6 | 零 |
| **L3** | **精度测试挡不住一致性 bug。** 确定性哈希(固定种子重复跑 + 顺序敏感哈希)抓到一个 wait 常量放松一档的 bug,**当时精度测试全绿** | 0.2 | `### Validating a Manual Correctness Argument` | 零 |
| **L4** | **融合不总省显存。** 对称堆 BF16 8.87 GiB;MXFP8 为 5.66 GiB 主缓冲 + 0.88 GiB signal pad = 6.54 GiB(约 BF16 的 3/4);每层省 0.766 GiB ⇒ 约第 12 层回本,4 层 proxy 仍净亏 | 0.2 | `### Activation Memory Saved` + `### Symmetric Heap Memory Spent` | 零 |
| **L6** | **liveness 是设计假设不是证明。** gfx950 被观测到大致按 block index 升序填 CU,但非架构保证;不钉 occupancy、不缩 grid(GEMM 区故意超订);MXFP8 combine 的在线搜索候选到 128 CU 时 margin 只剩填充顺序 | 0.3 | `### The Producer Prefix Is an Empirical Safeguard`(原文"It is not a proof.") | **需补 A7 压力实验**才能从免责声明升级为实证边界。**因为作者回应只有 3 天,这条必须在正文交代干净** |

---

## 图表预算

10 页双栏,**5 图全保留 + 主体 5 表**,其余合并或下放。

| 保留在主体 | 位置 |
|---|---|
| Figure 1 共轭前反向流水 | §4 |
| Figure 2 块角色特化 + 三张 signal 表的时序 | §5 |
| Figure 3 serial vs fused + 理想重叠 roofline | §7.2 |
| Figure 4 grouped GEMM vs 等效 FLOP 稠密 GEMM(红条保 192–240/256 CU) | §6 |
| Figure 5 两阶段 autograd 拆分前后的 profiler trace | §8 L1 |
| 表 1 两个算子各融了什么 | §4 |
| 表 2 实验设置 + 三个模型形状 | §6 |
| 表 3 BF16 五 stage 的 T_comm / XGMI / T_gemm / T_measured / speedup / η | §7.2 |
| 表 4 六模型训练步配对 | §7.4 |
| **表 5(新)实现机制合并表** = 三张 signal 表 + cache modifier 18/19 + CU 划分,压成一张 | §5 |

**下放附录**(注意审稿人无义务读,故不得承载核心结论):15 stage 全表、低精度五个子节的细节、MXFP8 分配器公式、复现命令。

**从 blog 整段删除**:`## Reproduce`(改为 artifact 附录)、`## Disclaimers`、以及全部指向未发布 companion post 的引用。

---

## 补什么:按"是否卡住论文"排序

| 优先级 | 项 | 为什么是这个优先级 | 对应 |
|---|---|---|---|
| **P0 行政** | 对外发表审批 | 不通则后面全是空的;且 blog 自身内部 review 未完(分支最近两个 commit 仍是 "fix review findings") | — |
| **P0 资格** | §3 部署刻画 | 决定论文有无资格投工业轨;**只需整理事实,不用跑实验**,应当最先做 | A3 |
| **P1 硬补** | §7.5 收敛曲线 | 训练系统论文的收敛证据无替代 | A1 = roadmap **T1-4** |
| **P1 硬补** | §7.6 不均衡路由步级 | **原论据引自未发布文章,不可引用** —— 这是搭框架时才发现的,优先级高于我上一轮的判断 | A2 |
| **P1 硬补** | §7.6 开 shared experts 配对 | blog 自己标注的偏向项,基准诚实度是工业轨核心标准 | A2 |
| **P1 零成本** | §8 L5 vendoring 教训 | 纯写作,证据已在仓里,差异化最强 | — |
| **P2 应做** | §7.7 Triton-distributed 前向 + DeepEP 算子级 | 任何轨都会问"有没有和你硬件上的开源替代比过" | A4 = roadmap **T3-3** |
| **P2 抬高度** | §8 L6 的 liveness 压力实验 | 3 天 rebuttal 意味着不能留白 | A7 |
| **P3 对冲** | gfx1250 第二平台 | 2028 年中开会时 MI355X 已两代前 | A6 |

---

## 核查溯源

本份的章节映射、全部图注与文件名、全部数字(46%/41%、1.42–1.54× / η 85.6–88.5% 五 stage、1.29–1.61× / η 82.7–94.2% 十五 stage、359–373 GB/s = 537.6 GB/s 的 67–69%、六模型 1.162–1.286× BF16 / 1.169–1.350× MXFP8、各模型深度与 MBS/GBS、8.87 GiB / 6.54 GiB = 5.66 + 0.88、每层 0.766 GiB、cache modifier 18/19、CU 划分 16/32/16 与 64/16、192–240 of 256 CU)均于 2026-10-28 从 `primus-mega-moe/README.md` 直接读取核对。

`#486` 的 grad norm 1.45 → 1.88e6 与 `f6d5ab68` 快照来自 `primus_turbo/flydsl/mega/bf16/__init__.py:14-24` 注释原文。Triton-distributed 的 `raise NotImplementedError` 与前向切分 docstring 来自本地只读检查。

**"companion post 未发布"这一点**由 `git ls-tree -d origin/develop blogs/software-tools-optimization/primus-moe-optimization` 核验为不存在,详见 [`2026-10-28_2050_megamoe-paper-plan.md`](./2026-10-28_2050_megamoe-paper-plan.md) §0。

**待核实**:MLSys 2028 实际 deadline(CFP 未发布)、CFP 的 Authorship Quota 具体篇数、AI 使用声明是否确为必需、Artifact Evaluation 是否确与录取决定无关、§9 中 MLSys 2026 各前作与本文的实际重叠程度(**必须逐篇精读**,目前仅基于标题与摘要级描述)。

**需你提供,我核不到**:§3 的全部部署事实。
