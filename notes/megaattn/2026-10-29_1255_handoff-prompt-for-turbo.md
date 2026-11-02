# Handoff prompt — MegaAttn 继续推进（给 Primus-Turbo / FlyDSL 会话）

> 复制下面「Prompt」整段，贴进新开的 Turbo / FlyDSL 工作区 agent。本文件本身是 handoff 存档，不要当论文大纲用。

---

## Prompt

```text
你在 AMD MI355X (gfx950 / CDNA4) 上推进 MegaAttn：把 DeepSeek Sparse Attention 的 decode 三段
（lightning indexer / paged MQA logits → top-k 2048 → sparse MLA）融进一个 persistent kernel，
复用 MegaMoE 已验证的 role-split 骨架。

# 必读（先读，再动手）

1. 立项与状态：
   `/perf_apps/xiaoming/slab/notes/megaattn/README.md`
2. v1 架构设计（瓶颈估算、与 MegaMoE 同构映射、M0–M4、风险表）：
   `/perf_apps/xiaoming/slab/notes/megaattn/2026-08-05_1650_megaattn_v1_architecture_design.md`
3. MegaMoE 骨架参考（role-split / signal table / work queue）：
   FlyDSL `kernels/mega_moe/` 或 Primus-Turbo 里对应的 `flydsl/mega/` 路径
4. 已有 MLA decode 资产（缺的是「按索引 gather」，不是 dense tile loop）：
   FlyDSL `kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py`
5. 稀疏路径被硬挡：`pa_metadata.py` 里 `assert topk == -1`

工作仓优先：`~/workspace/FlyDSL/kernels/attention/`（若当前是 Primus-Turbo 树，先定位 attention / mega 等价路径，不要假设路径名）。
对拍基准：DeepGEMM / MegaMOE 的 paged MQA logits 与 top-k 语义（设计文档里写的参考路径）。

# 当前状态（2026-10-29）

- 设计成文于 2026-08-05；**M0 摸底从未启动**。不要直接写融合 kernel。
- 唯一允许的第一步是 **M0：三独立 kernel 基线 + 分段 profile**。

# 业界已变（相对 8 月设计，必须按这个更新判断）

| 工作 | 出处 | 含义 |
|---|---|---|
| LiteTopK + LiteDSA | arXiv:2607.11976（v3）；vLLM #48726 | **Indexer–TopK 融合已被正式发文并开源进 vLLM**。你们设计里「在线阈值 + 不落全量 logits」不再是空白。LiteDSA 另做相邻 query KV packing。 |
| GVR | arXiv:2604.22312（原 NVIDIA 技术报告已成文） | decode Top-K 的跨 step 时序复用已被占；精确 top-k，算子 ~1.88×。 |
| SnapMLA | arXiv:2602.10718 | FP8 MLA decode 数值方案；可借，不是三段融合。 |
| HISA | arXiv:2603.28458 | 层次 indexer（算法改搜索路径），不是 megakernel。 |

**未看到的**：把 indexer + top-k + sparse MLA **三段**做成 MegaMoE 式 role-split persistent grid 的论文。
**AMD / CDNA4 上**：仍未见同等融合实现。

LiteTopK 自己报 decode 端到端最高约 **1.04×**（Indexer–TopK 只占一小段）——这直接坐实设计文档 **R1**：若 top-k + logits 往返 wall < 15%，融合天花板太低，应叫停。

# 硬约束

1. **先 M0，再谈 M1/M2/M3。** M0 没出占比表前，禁止开融合实现或大改 FlyDSL 稀疏接口。
2. M0 验收（设计文档原门槛，不要放宽）：
   - 平台：MI355X / gfx950，在 `xiaoming-dev` 容器 + squeue 节点上跑，不要在 login 机上编跑。
   - N = 8K / 32K / 128K 三档（至少）。
   - 分段量：**indexer / top-k / sparse MLA** 各自 wall 与占比；同时记录 logits 写回 + top-k 多趟读的合计。
   - **决策**：若「top-k + logits 往返」合计 < 15% wall → **就地叫停**，写 note 记录数据与结论，然后评估设计文档 §7 备选（优先：Attention 通信融合 / CP 方向；其次 Mega-MLA；再次 FMHA bwd）。不要硬做 M2。
3. 精确性：后续任何 top-k 融合路径必须可对拍到**精确 top-k**（与 LiteTopK / GVR 同档），不要先上近似 top-k 糊弄过门。
4. 范围：MegaAttn v1 **只做前向**；backward 不在本轮。
5. 不要跟：通用 megakernel 编译器（MPK / Event Tensor / AutoMegaKernel）。
6. 笔记写回：`/perf_apps/xiaoming/slab/notes/megaattn/`，文件名 `YYYY-MM-DD_HHMM_<topic>.md`；更新 `README.md` 状态表。不要把发现写进 `.cursor/skills/`。
7. 未经明确指示不要 git commit。

# 本轮交付物（只做这些）

1. **M0 可复现实验脚本 / 命令**：三独立 kernel 基线，N=8K/32K/128K，rocprof 或等价分段计时。
2. **一张占比表**（每档 N × 三段 wall + 占比 + logits 往返估计）。
3. **GO / NO-GO 结论**：相对 15% 门槛写清楚；若 NO-GO，按设计文档 §7 给出下一备选与理由（一页内）。
4. **简短进度 note** 写入 `notes/megaattn/`，并更新 README 状态行。

若 M0 通过（≥15%），下一轮才允许开 M1（独立 FP8 paged MQA logits + k=2048 radix top-k，对拍 DeepGEMM）。**本轮不要提前做 M1 代码。**

# 写作与工程风格

- 数字必须来自本次实测或已引用文献；不要发明硬件规格、路径、加速比。
- 对比 LiteTopK / GVR 时说清：我们比的是 **CDNA4 + 三段一核**，不是再发明一遍 Indexer–TopK 融合叙事。
- 优先表格；路径用反引号。
```

---

## 使用说明

- 在 **FlyDSL 或 Primus-Turbo** 工作区开新 agent，整段粘贴即可。
- 若 agent 只能看到 Turbo 树、看不到 slab：把 prompt 里的绝对路径改成它能读到的挂载路径，或先把 `notes/megaattn/` 两份 md 贴进上下文。
- 本轮故意收窄成 **只做 M0**；避免 agent 一上来就写融合 kernel 或跟 LiteTopK 正面撞车。
