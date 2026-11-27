# Experiment registry

| ID | Question | Comparison |
|---|---|---|
| E1 | Transport cost | Latency, bandwidth, message size, concurrent compute |
| E2 | Tile readiness benefit | Separate kernels vs coarse barrier vs fine readiness |
| E3 | Fusion boundary | Projection vs attention fusion; compact vs expanded representation |
| E4 | MoE overlap | Dispatch/GEMM/combine, forward and backward, skew |
| E5 | Heterogeneous transport | CU vs copy engine vs hybrid, same workload |
| E6 | End-to-end | Baseline, CP optimization, EP optimization, both |

Record code revision, environment, topology, shape, precision, baseline, warmup, repetitions, statistic, correctness tolerance, memory, raw evidence reference and limitations. Keep failed runs. Use only approved public evidence here; private raw logs stay in authorized environments.
