# Team Weekly Report (11/23 - 11/29)

> **When**: 2026-11-27 08:46 UTC+8
> **Where**: compiled by Peng, Xiaoming from team updates
> **Context**: team weekly; paste-ready body below the rule

---

MI455X Primus/Turbo v6 released (led by Zhang, Lihuan)

- Released dev/mi455x_v6 in [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) and [Primus](https://github.com/AMD-AGI/Primus): v6 merged into main on 11-25 (Primus PR #1527, Primus-Turbo PR #685; image amdprimus/amdprimus:tas-20261109), together with the MI455-related code refactoring. It adds Muse Spark grouped GEMM retuning for EP4–EP32 (+8.4% geomean, bit-identical output), the FlyDSL SiTU kernel with moe_router_fusion for Kimi K3.1 Pro, and bitwise Triton / FlyDSL kernels with a BF16 recipe for MiniMax-M3 / M3.1-Flash.
- P0 model optimization (single GPU; there was no usable 1P4G system this week, so multi-GPU performance is still to be tested)
  - Kimi K3.1 Pro: 30.4k → 34.9k tok/s/GPU (+15%), peak memory 383.8 → 343.0 GB (−40.8 GB) - Yang, Bo
  - MiniMax-M3 / M3.1-Flash: 57.7k → 88.8k tok/s/GPU (+54%, 1.54×), peak memory 199.9 → 116.7 GB (−83.2 GB) - Huang, Wei
  - DeepSeek-V3: MXFP8 linears and routed experts are under review, waiting for 1P4G tests - Zhang, Lihuan
- v7 development continues on dev/mi455x_v7 in [Primus](https://github.com/AMD-AGI/Primus/tree/dev/mi455x_v7) and [Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo/tree/dev/mi455x_v7).

Muse Spark Grouped GEMM - Rao, Lorri

- Tuned the Muse Spark grouped GEMM for EP4–EP32 on MI455X (Primus-Turbo PR #654): +8.4% geomean, with output bit-identical to the untuned path.
- Retuned the keys on a production MI455X node. The untuned kernels run 1.2–2.1× faster there, so the earlier Muse Spark keys (#640) measured −0.52% against the untuned path; retuned in step mode for EP4 / EP8 at MBS 1 / 2.
- Six-op step, keys on vs off at MBS 1 / 2: EP4 +6.5% / +8.1%, EP8 +10.5% / +8.2%, EP16 +8.0% / +7.8%, EP32 +11.7% / +8.0%. The six GG operators combined gain +6.5–11.7% across 10 configs, and row counts between keys gain +7.6–9.1%.
- Added nearest avg_m matching. Under expert parallelism the rows a GPU receives vary per step, so the exact (N, K, avg_m) lookup missed; Muse Spark shapes now take the nearest keyed avg_m on a log scale (within √2 of the key range), and other models' keys stay exact.
- Added EP16 / EP32 keys for 4096 / 8192 rows per expert.

Grouped GEMM - Hu, Botao

- Tuned FlyDSL grouped GEMM for the MiniMax-M3.1-Flash and Kimi K3.1 Pro shapes. All six shapes meet the ≥85% of MAIDAS requirement at the same frequency (85.7–94.4%).
- MiniMax-M3.1-Flash (N=6144, K=6144): true steady state rose from 1887.7 to 2316.8 TF/s at M=512 (+22.7%, 94.0% of MAIDAS), from 2185.2 to 2689.1 TF/s at M=1024 (+23.1%, 93.3%), and from 2328.5 to 2809.2 TF/s at M=2048 (+20.6%, 93.6%).
- Kimi K3.1 Pro (N=6144, K=3584): true steady state rose from 1272.0 to 1785.5 TF/s at M=293 (+40.4%, 89.2% of MAIDAS), from 1722.5 to 2171.7 TF/s at M=585 (+26.1%, 85.7%), and from 1930.6 to 2419.4 TF/s at M=1170 (+25.3%, 94.4%).

Attention - Zhang, Lihuan

- [Ongoing] Implemented Muse Spark SWA attention on FlyDSL, 3.6–4.6× faster than Triton; end-to-end tests are next.

Kimi K3.1 Pro / DeepEP / MoonEP - Yang, Bo

- Kimi K3.1 Pro: Merged PR #1451 (FlyDSL SiTU with the router probability folded in + moe_router_fusion) and PR #1526 into mi455x_v6. SiTU routed experts (fwd + bwd) dropped from 16.26 to 0.91 ms (17.9×), and single-GPU throughput rose from 30.4k to 34.9k tok/s/GPU (+15%). Next: KDA attention projection.
- DeepEP: Reported TDM loads lost on queue eviction as PLAT-218497; the driver team triaged it as blocked by PLAT-216134 (same symptom).
- MoonEP: Worked on MoonEP / UltraEP for the Turbo MoE layers on MI355X. On a DeepSeek-V3 4-layer proxy at EP8 with real routing, step time is 3% faster. Next: EP64-like imbalance (one-node simulation) and synthetic skew.

MiniMax-M3 / M3.1-Flash - Huang, Wei

- Landed Triton fusion and FlyDSL MSA optimization (PR #1464). MSA reached 578 TF/s at 4k sequence length, and the single-GPU proxy rose from 57.7k to 88.8k tok/s/GPU (+54%).
- [Ongoing] Further FlyDSL MSA optimization.

All-gather KV CP Attention (MI355X) - Peng, Xiaoming

- Fused the KV all-gather and the gradient reduce-scatter of all-gather KV context-parallel attention into persistent FlyDSL kernels, integrated them into Primus, and validated end-to-end on Llama3.1-8B and Qwen3-30B.
