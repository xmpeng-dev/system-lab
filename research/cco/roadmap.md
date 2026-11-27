# CCO research roadmap

Proposed research milestones, not employer commitments. All statuses are planning defaults, not claims of implementation.

| ID | Horizon | Workstream | Exit criterion |
|---|---|---|---|
| C1 | Near term | Transport and synchronization | Reproducible latency/bandwidth and correctness tests |
| C2 | Near term | Collective GEMM | Fair AG-GEMM and GEMM-RS baselines, then A2A variants |
| C3 | Near term | Expert parallelism | Forward/backward correctness and end-to-end comparison |
| C4 | Medium term | Context parallelism | Compare projection fusion and attention-level fusion |
| C5 | Medium term | Heterogeneous overlap | Compare compute-unit and copy-engine paths where supported |
| C6 | Medium term | Research publication | Evidence ledger, ablations, limitations and reproducibility |
| C7 | Long term | Adaptive scheduling | Validated policy across shapes, topology and precision |

Start with BF16 reference implementations; extend precision only after correctness and baseline quality are established. Evaluate scale-up first, then larger domains where resources and authorization permit. Do not assume device-triggered copy support.
