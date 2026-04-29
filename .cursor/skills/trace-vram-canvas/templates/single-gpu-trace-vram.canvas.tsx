/**
 * TEMPLATE — single-rank trace + VRAM canvas.
 * See `.cursor/skills/trace-vram-canvas/SKILL.md` for the workflow.
 *
 * Works for any single-node training run regardless of fine-tuning mode
 * (LoRA SFT / full SFT / pretraining / MoE) and regardless of vendor
 * (AMD MI300X..MI355X / NVIDIA H100 / B200). Replace the constants in the
 * "Run identity" / "Kernel categories" / "Per-stream busy" / "Pipeline
 * lanes" / "VRAM" / "VRAM_BUCKETS" sections below. The SVG components
 * (`PipelineDiagram`, `Legend`, `VRAMBar`, `StackedBucketBar`) and the
 * palette stay as-is.
 *
 * Source data the constants come from:
 *   - trace: <path>/torch_profiler_traces/<host>.<pid>.pt.trace.json
 *   - run.log: <path>/<runname>/<runname>.log
 *   - breakdown: output of full_breakdown.py on a steady-state ProfilerStep
 *
 * IMPORTANT on FP8 runs: re-categorize FP8 GEMM kernels — the analyzer
 * puts them in `other` by default. See SKILL.md Step 3 for the kernel
 * name patterns per backend (AMD hipBLASLt / CK-tile / NVIDIA cuBLASLt /
 * TE cutlass).
 *
 * For LoRA runs use the Mode A bucket set in `VRAM_BUCKETS`; for full FT
 * use Mode B; for MoE pretraining use Mode C. See SKILL.md Step 5.
 */
import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/* ── Run identity ─────────────────────────────────────────────────────── */
const RUN = {
  hardware: "8 × MI355X (288 GiB HBM)",
  model: "Llama-2-70B · LoRA SFT · bf16+fp8 hybrid",
  parallelism: "TP1 PP1 CP1 EP1 — pure DP=8 (DDP + DistOpt)",
  batch: "GBS 8 · MBS 1 · seq 8192 (packed)",
  step: "ProfilerStep #82",
  stepMs: 1626.45,
  samplesPerStep: 8,
  tflopsPerGPU: 2251.0,
  trainable: "44.5M / 69.0B (0.06%)",
};

/* ── Kernel categories (re-categorized: FP8 Custom_Cijk_* belong to GEMM) ─ */
type KernelRow = { category: string; ms: number; tone?: "info" | "warning" | "success" };
const KERNEL_ROWS: KernelRow[] = [
  { category: "FP8 GEMM (Custom_Cijk_*_F8*, hipBLASLt)", ms: 773.5, tone: "info" },
  { category: "FlashAttention (aiter::fmha fwd+bwd)", ms: 320.2 },
  { category: "Elementwise / cast / dropout (vectorized_*)", ms: 122.8 },
  { category: "TE SwiGLU / dgated_act / unary (gated_act_kernel*)", ms: 64.1 },
  { category: "RCCL grad-sync (ncclDevKernel_Generic_1)", ms: 54.4, tone: "warning" },
  { category: "bf16 GEMM (Cijk_Ailk_Bljk_BBS / BSS)", ms: 38.5 },
  { category: "RMSNorm (triton fwd+bwd)", ms: 20.3 },
  { category: "Fused QKV-RoPE (fwd+bwd)", ms: 23.3 },
  { category: "FP8 cast / transpose (cast_transpose, transpose_optimized)", ms: 21.2 },
  { category: "Reduction kernels", ms: 8.6 },
  { category: "MemCopy / D2D", ms: 4.5 },
  { category: "Optimizer / softmax / misc", ms: 0.3 },
];

const KERNEL_TOTAL_MS = KERNEL_ROWS.reduce((a, r) => a + r.ms, 0);

/* ── Top-10 individual kernel names (from breakdown) ───────────────────── */
const TOP_KERNELS: Array<{ name: string; ms: number; pct: number; bucket: string }> = [
  { name: "Custom_Cijk_Alik_Bljk_F8B8BS …shortname1 (FP8 GEMM)", ms: 350.43, pct: 21.5, bucket: "GEMM fwd" },
  { name: "Custom_Cijk_Alik_Bljk_F8BS …shortname1 (FP8 GEMM)",   ms: 279.88, pct: 17.2, bucket: "GEMM bwd" },
  { name: "aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv",         ms: 212.53, pct: 13.1, bucket: "Attention bwd" },
  { name: "Custom_Cijk_Alik_Bljk_F8BS …shortname0 (FP8 GEMM)",   ms: 104.18, pct: 6.4,  bucket: "GEMM" },
  { name: "aiter::fmha_fwd_hd128_bf16_causal",                    ms:  90.57, pct: 5.6,  bucket: "Attention fwd" },
  { name: "ncclDevKernel_Generic_1 (RCCL grad sync)",             ms:  54.39, pct: 3.3,  bucket: "Collective" },
  { name: "vectorized_elementwise_kernel<CUDAFunctor_add bf16>", ms:  43.40, pct: 2.7,  bucket: "Elementwise" },
  { name: "Custom_Cijk_Alik_Bljk_F8B8BS …shortname0 (FP8 GEMM)", ms:  38.96, pct: 2.4,  bucket: "GEMM" },
  { name: "transformer_engine::gated_act_kernel<silu>",           ms:  34.10, pct: 2.1,  bucket: "Activation" },
  { name: "transformer_engine::dgated_act_kernel<silu>",          ms:  30.52, pct: 1.9,  bucket: "Activation" },
];

/* ── Per-stream busy ──────────────────────────────────────────────────── */
const STREAM_ROWS: Array<{
  stream: string;
  role: string;
  busyMs: number;
  share: string;
}> = [
  { stream: "stream 0 (pid 2)",  role: "Compute (GEMM / FMHA / norm / elem / activation)", busyMs: 1377.7, share: "84.7%" },
  { stream: "stream 33 (pid 2)", role: "RCCL DDP grad-sync (Generic_1)",                   busyMs:   54.4, share: " 3.3%" },
];

/* ── Pipeline lanes (one ProfilerStep, normalized to 100) ─────────────── */
type SegKind =
  | "gemm" | "attn" | "norm" | "moe" | "elem" | "opt"
  | "a2a" | "rs" | "ag" | "ar" | "idle";
type Seg = { kind: SegKind; t: number; w: number; label?: string };

// Bin layout from breakdown_step82 (each bin = 20.3 ms = 1.25% of step):
//   bins 0-8   (0-180 ms,  ~11%): warmup / data loader / dispatcher → idle
//   bins 9-37  (180-770 ms, ~36%): forward (FP8 GEMM + attn fwd + elem)
//   bin  38    (770-790 ms,  ~1%): fwd↔bwd transition
//   bins 39-76 (790-1565 ms, ~48%): backward (attn bwd + FP8 GEMM bwd)
//   bins 77-79 (1565-1626 ms, ~4%): pure RCCL (gradient sync, serial!)
const LANES: Array<{ name: string; tag: string; segs: Seg[] }> = [
  {
    name: "stream 0",
    tag: "compute",
    segs: [
      { kind: "idle", t: 0,   w: 11 },
      { kind: "gemm", t: 11,  w: 13, label: "FP8 GEMM fwd" },
      { kind: "attn", t: 24,  w: 6,  label: "fmha fwd" },
      { kind: "elem", t: 30,  w: 4 },
      { kind: "norm", t: 34,  w: 1 },
      { kind: "gemm", t: 35,  w: 12, label: "FP8 GEMM fwd" },
      { kind: "attn", t: 47,  w: 1 },
      { kind: "gemm", t: 48,  w: 14, label: "FP8 GEMM bwd" },
      { kind: "attn", t: 62,  w: 13, label: "fmha bwd" },
      { kind: "gemm", t: 75,  w: 13, label: "FP8 GEMM bwd" },
      { kind: "elem", t: 88,  w: 5 },
      { kind: "opt",  t: 93,  w: 3,  label: "step" },
      { kind: "idle", t: 96,  w: 4 },
    ],
  },
  {
    name: "stream 33",
    tag: "RCCL DDP",
    segs: [
      { kind: "idle", t: 0,  w: 96 },
      { kind: "ar",   t: 96, w: 4, label: "grad sync (serial!)" },
    ],
  },
];

const COMPUTE_BUSY_PCT = 84.7;
const NCCL_HIDDEN_PCT = 0.4;
const STREAM_OVERSUB_PCT = 88;
const IDLE_MS = 191.6;

/* ── VRAM (8 × MI355X, per-GPU = 309.22 GB / 287.99 GiB) ──────────────── */
const VRAM = {
  capGB: 309.22,         // totalGlobalMem from trace deviceProperties
  capGiB: 287.99,
  pmax: 285.84,          // mem-max-allocated-gigabytes (peak working set)
  rmax: 295.52,          // mem-max-reserved-gigabytes (driver-side cached peak)
  current: 126.44,       // mem-allocated-gigabytes at iter 10
  retires: 0,
};
const RESERVED_PCT = (VRAM.rmax / VRAM.capGB) * 100;
const ALLOC_PCT = (VRAM.pmax / VRAM.capGB) * 100;
const FRAG_PCT = ((VRAM.rmax - VRAM.pmax) / VRAM.rmax) * 100;
const HEADROOM = VRAM.capGB - VRAM.rmax;

// Estimated bucket split (calibrated to Pmax=285.8 GB).
// LoRA: trainable params (44.5M) → grads/opt are tiny.
// bf16+fp8 hybrid: weights stored partly in fp8 → ~120 GB.
// Activations dominate at seq=8192, no recompute, packed sequences.
const VRAM_BUCKETS: Array<{ bucket: string; gb: number; note: string }> = [
  { bucket: "Weights (bf16 + fp8 hybrid)",         gb: 120.0, note: "70B params, mixed precision" },
  { bucket: "Activations (no recompute)",          gb: 145.0, note: "seq=8192 × layers=80 × bf16" },
  { bucket: "LoRA grads + Adam state (44.5M trainable)", gb:  0.8, note: "tiny — distributed across DP=8" },
  { bucket: "TE FP8 caches / cuBLAS workspace / NCCL bufs", gb: 12.0, note: "~10-15 GB typical" },
  { bucket: "Allocator slack (Rmax − Pmax)",       gb: VRAM.rmax - VRAM.pmax, note: "fragmentation" },
  { bucket: "Other / unaccounted",                 gb: VRAM.pmax - 120.0 - 145.0 - 0.8 - 12.0, note: "sanity gap vs Pmax" },
];

/* ── Palette ──────────────────────────────────────────────────────────── */
const SEG_COLORS: Record<SegKind, string> = {
  gemm: "#2E79B5E0",
  attn: "#7B64B8F0",
  norm: "#70B0D8E0",
  moe:  "#F0A040E0",
  elem: "#8888A8E0",
  opt:  "#1F8A65E8",
  a2a:  "#C04848E0",
  rs:   "#E8C030E0",
  ag:   "#7DCAB0E0",
  ar:   "#C85898E0",
  idle: "transparent",
};

function useSegPalette() {
  const t = useHostTheme();
  return { ...SEG_COLORS, idle: t.fill.tertiary } as Record<SegKind, string>;
}

function PipelineDiagram({
  totalMs,
  lanes,
  width = 760,
}: {
  totalMs: number;
  lanes: Array<{ name: string; tag: string; segs: Seg[] }>;
  width?: number;
}) {
  const t = useHostTheme();
  const palette = useSegPalette();
  const labelW = 160;
  const trackH = 26;
  const gap = 8;
  const trackW = width - labelW;
  const height = lanes.length * (trackH + gap) + 28;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {[0, 25, 50, 75, 100].map((p) => {
        const x = labelW + (trackW * p) / 100;
        return (
          <g key={p}>
            <line
              x1={x} x2={x} y1={0}
              y2={lanes.length * (trackH + gap)}
              stroke={t.stroke.tertiary}
              strokeDasharray="2 3"
            />
            <text x={x} y={lanes.length * (trackH + gap) + 16}
              fontSize={10} fill={t.text.quaternary} textAnchor="middle">
              {((p / 100) * totalMs).toFixed(0)} ms
            </text>
          </g>
        );
      })}
      {lanes.map((lane, i) => {
        const y = i * (trackH + gap);
        return (
          <g key={lane.name}>
            <text x={0} y={y + trackH / 2 + 4} fontSize={11}
              fill={t.text.secondary}
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace">
              {lane.name}
            </text>
            <text x={80} y={y + trackH / 2 + 4} fontSize={10} fill={t.text.quaternary}>
              {lane.tag}
            </text>
            <rect x={labelW} y={y} width={trackW} height={trackH}
              fill={t.fill.quaternary} rx={3} />
            {lane.segs.map((s, j) => {
              const sx = labelW + (trackW * s.t) / 100;
              const sw = (trackW * s.w) / 100;
              const isIdle = s.kind === "idle";
              return (
                <g key={j}>
                  <rect x={sx} y={y} width={sw} height={trackH}
                    fill={isIdle ? "transparent" : palette[s.kind]}
                    opacity={isIdle ? 0 : 0.92} rx={2}>
                    <title>{`${s.kind}${s.label ? " · " + s.label : ""} (${s.w}% · ${(s.w * totalMs / 100).toFixed(1)} ms)`}</title>
                  </rect>
                  {s.label && sw > 50 ? (
                    <text x={sx + 4} y={y + trackH / 2 + 3} fontSize={9}
                      fill={t.text.onAccent ?? "#fff"}
                      style={{ pointerEvents: "none" }}>
                      {s.label}
                    </text>
                  ) : null}
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function Legend() {
  const palette = useSegPalette();
  const items: Array<{ kind: SegKind; label: string }> = [
    { kind: "gemm", label: "GEMM (FP8 + bf16)" },
    { kind: "attn", label: "FlashAttention" },
    { kind: "norm", label: "RMSNorm" },
    { kind: "elem", label: "Elementwise / SwiGLU" },
    { kind: "opt",  label: "Optimizer step" },
    { kind: "ar",   label: "RCCL grad sync (serial)" },
  ];
  return (
    <Row gap={12} wrap>
      {items.map((it) => (
        <Row key={it.kind} gap={6} align="center">
          <span style={{
            display: "inline-block", width: 12, height: 12,
            background: palette[it.kind], borderRadius: 2,
          }} />
          <Text size="small" tone="secondary">{it.label}</Text>
        </Row>
      ))}
    </Row>
  );
}

function VRAMBar({ capGB, pmax, rmax }: { capGB: number; pmax: number; rmax: number }) {
  const t = useHostTheme();
  const palette = useSegPalette();
  const W = 720, H = 36;
  const wmax = (rmax / capGB) * W;
  const wpmax = (pmax / capGB) * W;
  const wcap = W;
  return (
    <svg width={W} height={H + 18} style={{ display: "block" }}>
      <rect x={0} y={0} width={wcap} height={H} rx={4} fill={t.fill.tertiary} />
      <rect x={0} y={0} width={wmax}  height={H} rx={4} fill={palette.rs} opacity={0.55}>
        <title>{`Reserved peak: ${rmax.toFixed(1)} GB (${(rmax/capGB*100).toFixed(1)}% of cap)`}</title>
      </rect>
      <rect x={0} y={0} width={wpmax} height={H} rx={4} fill={palette.gemm} opacity={0.85}>
        <title>{`Allocated peak: ${pmax.toFixed(1)} GB (${(pmax/capGB*100).toFixed(1)}% of cap)`}</title>
      </rect>
      <line x1={wcap} x2={wcap} y1={0} y2={H} stroke={t.stroke.primary} strokeWidth={2} />
      <text x={wpmax - 6} y={H/2 + 4} fontSize={11} fill={t.text.onAccent ?? "#fff"} textAnchor="end">
        Pmax {pmax.toFixed(1)} GB
      </text>
      <text x={wmax + 6} y={H/2 + 4} fontSize={11} fill={t.text.secondary}>
        Rmax {rmax.toFixed(1)} GB
      </text>
      <text x={wcap - 4} y={H + 14} fontSize={10} fill={t.text.tertiary} textAnchor="end">
        cap {capGB.toFixed(1)} GB
      </text>
    </svg>
  );
}

function StackedBucketBar({ buckets, totalGB }: { buckets: typeof VRAM_BUCKETS; totalGB: number }) {
  const t = useHostTheme();
  const palette = useSegPalette();
  const colorOrder: SegKind[] = ["gemm", "attn", "elem", "norm", "rs", "moe"];
  const W = 720, H = 32;
  let cursor = 0;
  return (
    <svg width={W} height={H + 14} style={{ display: "block" }}>
      <rect x={0} y={0} width={W} height={H} rx={4} fill={t.fill.tertiary} />
      {buckets.map((b, i) => {
        const w = (Math.max(0, b.gb) / totalGB) * W;
        const color = palette[colorOrder[i % colorOrder.length]];
        const node = (
          <g key={b.bucket}>
            <rect x={cursor} y={0} width={w} height={H} fill={color} opacity={0.85}>
              <title>{`${b.bucket}: ${b.gb.toFixed(1)} GB (${((b.gb/totalGB)*100).toFixed(1)}% of Pmax)`}</title>
            </rect>
            {w > 80 ? (
              <text x={cursor + 6} y={H/2 + 4} fontSize={10}
                fill={t.text.onAccent ?? "#fff"} style={{ pointerEvents: "none" }}>
                {b.bucket.split(" (")[0]}
              </text>
            ) : null}
          </g>
        );
        cursor += w;
        return node;
      })}
      <text x={W - 4} y={H + 12} fontSize={10} fill={t.text.tertiary} textAnchor="end">
        sum = Pmax {totalGB.toFixed(1)} GB
      </text>
    </svg>
  );
}

export default function Llama2BaselineTrace() {
  const msPerSample = RUN.stepMs / RUN.samplesPerStep;
  const samplesPerSec = (RUN.samplesPerStep * 1000) / RUN.stepMs;
  const reservedTone: "success" | "warning" | "danger" =
    RESERVED_PCT >= 98 ? "danger" : RESERVED_PCT >= 95 ? "warning" : "success";

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>{RUN.hardware} — Llama-2-70B LoRA SFT baseline trace</H1>
        <Text tone="secondary">{RUN.model}</Text>
        <Row gap={8} wrap>
          <Pill tone="info">{RUN.step}</Pill>
          <Pill tone="neutral">{RUN.parallelism}</Pill>
          <Pill tone="neutral">{RUN.batch}</Pill>
          <Pill tone="neutral">{RUN.trainable}</Pill>
        </Row>
      </Stack>

      <Grid columns={5} gap={12}>
        <Stat value={`${RUN.stepMs.toFixed(0)} ms`} label="step time" />
        <Stat value={`${RUN.tflopsPerGPU.toFixed(0)}`} label="TFLOP/s/GPU" tone="success" />
        <Stat value={`${samplesPerSec.toFixed(2)}`} label="samples/sec (8 GPU)" />
        <Stat value={`${COMPUTE_BUSY_PCT.toFixed(0)}%`}
          label="compute busy"
          tone={COMPUTE_BUSY_PCT > 80 ? "success" : "warning"} />
        <Stat value={`${NCCL_HIDDEN_PCT.toFixed(1)}%`}
          label="NCCL hidden"
          tone="danger" />
      </Grid>

      <Divider />

      <H2>Per-stream timeline (one ProfilerStep #82)</H2>
      <Text tone="secondary">
        Two streams active: stream 0 carries all compute (1377.7 ms = 84.7% busy),
        stream 33 carries the gradient all-reduce (54.4 ms, runs entirely after
        backward — <Text as="span" weight="semibold">0.4% overlap with compute</Text>).
        About 191 ms (~12%) is idle, mostly the data-loader/dispatcher gap at step start.
      </Text>
      <Legend />
      <PipelineDiagram totalMs={RUN.stepMs} lanes={LANES} />

      <Divider />

      <H2>GPU work decomposition by kernel category</H2>
      <Text tone="secondary">
        Sum of busy time per kernel category across both streams. FP8 GEMMs
        (Custom_Cijk_*_F8*) account for almost half of the step on their own;
        FlashAttention (aiter::fmha) is the next biggest bucket.
      </Text>
      <Grid columns={2} gap={16}>
        <Table
          headers={["Category", "ms", "% of step"]}
          columnAlign={["left", "right", "right"]}
          rows={KERNEL_ROWS.map((r) => [
            r.category,
            r.ms.toFixed(1),
            `${(r.ms / RUN.stepMs * 100).toFixed(1)}%`,
          ])}
          rowTone={KERNEL_ROWS.map((r) => r.tone)}
        />
        <Stack gap={8}>
          <Text size="small" tone="secondary">
            Total accounted: {KERNEL_TOTAL_MS.toFixed(1)} ms ({(KERNEL_TOTAL_MS / RUN.stepMs * 100).toFixed(1)}% of step) ·
            idle/gap: {IDLE_MS.toFixed(1)} ms · oversub {STREAM_OVERSUB_PCT}%
          </Text>
          <Table
            headers={["Stream", "Role", "ms", "Share"]}
            columnAlign={["left", "left", "right", "right"]}
            rows={STREAM_ROWS.map((r) => [r.stream, r.role, r.busyMs.toFixed(1), r.share])}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>Top-10 individual kernels by duration</H2>
      <Table
        headers={["Kernel", "Bucket", "ms", "% of step"]}
        columnAlign={["left", "left", "right", "right"]}
        rows={TOP_KERNELS.map((k) => [k.name, k.bucket, k.ms.toFixed(1), `${k.pct.toFixed(1)}%`])}
      />

      <Divider />

      <H2>VRAM (HBM) usage — rank 0, MI355X 288 GiB cap</H2>
      <Text tone="secondary">
        Source: <Text as="span">train_utils.py:671</Text> after iter 10.
        Cap from trace <Text as="span">deviceProperties[*].totalGlobalMem = 309.22 GB</Text>.
      </Text>

      <Grid columns={4} gap={12}>
        <Stat
          value={`${RESERVED_PCT.toFixed(1)}%`}
          label="Reserved (driver-side)"
          tone={reservedTone} />
        <Stat
          value={`${ALLOC_PCT.toFixed(1)}%`}
          label="Allocated (working set)"
          tone={ALLOC_PCT >= 90 ? "warning" : "success"} />
        <Stat
          value={`${HEADROOM.toFixed(1)} GB`}
          label="Headroom to OOM"
          tone={HEADROOM < 15 ? "warning" : "success"} />
        <Stat
          value={`${VRAM.retires}`}
          label="Allocator retires"
          tone={VRAM.retires === 0 ? "success" : "warning"} />
      </Grid>

      <Stack gap={6}>
        <Text size="small" tone="secondary">
          Capacity bar — Pmax (allocated peak) inside Rmax (reserved peak) inside cap.
        </Text>
        <VRAMBar capGB={VRAM.capGB} pmax={VRAM.pmax} rmax={VRAM.rmax} />
      </Stack>

      <Stack gap={6}>
        <Text size="small" tone="secondary">
          Bucket decomposition (estimated, calibrated to Pmax = {VRAM.pmax.toFixed(1)} GB).
        </Text>
        <StackedBucketBar buckets={VRAM_BUCKETS} totalGB={VRAM.pmax} />
        <Table
          headers={["Bucket", "GB", "% of Pmax", "Note"]}
          columnAlign={["left", "right", "right", "left"]}
          rows={VRAM_BUCKETS.map((b) => [
            b.bucket,
            b.gb.toFixed(1),
            `${(b.gb / VRAM.pmax * 100).toFixed(1)}%`,
            b.note,
          ])}
        />
      </Stack>

      <Divider />

      <Card>
        <CardHeader trailing={<Pill tone="warning">3 actionable findings</Pill>}>
          What the trace tells us
        </CardHeader>
        <CardBody>
          <Stack gap={10}>
            <Text>
              <Text as="span" weight="semibold">1. Compute-bound on FP8 GEMM (49.9% of step).</Text>{" "}
              Four <Text as="span">Custom_Cijk_Alik_Bljk_F8*</Text> kernels eat
              773.5 ms — these are the LLaMA linear_qkv / linear_proj /
              gate_up_proj / down_proj fwd+bwd in fp8 hybrid recipe.
              FlashAttention (<Text as="span">aiter::fmha</Text> fwd+bwd) adds
              another 320 ms (19.7%). Together they are ~70% of the step, which
              is healthy for a 70B model at TFLOP/s/GPU = {RUN.tflopsPerGPU}.
            </Text>
            <Text>
              <Text as="span" weight="semibold">2. NCCL is fully serial (biggest opportunity).</Text>{" "}
              The gradient sync (<Text as="span">ncclDevKernel_Generic_1</Text>,
              54.4 ms) lands entirely in the last 60 ms of the step on stream 33;
              only 0.2 ms overlap with compute → 0.4% hidden. Enabling DDP comm
              overlap (<Text as="span">overlap_grad_reduce / overlap_param_gather</Text>)
              would reclaim up to ~3% of step time at zero risk.
            </Text>
            <Text>
              <Text as="span" weight="semibold">3. Step has a 191 ms idle gap (~12%) before compute starts.</Text>{" "}
              Bins 0-8 of the time-binned mix are empty — likely
              data-loader / dispatcher / first kernel launch latency. Worth
              checking <Text as="span">num_workers</Text> and whether a
              <Text as="span"> prefetch_factor</Text> bump or persistent workers
              can shrink it.
            </Text>
            <Callout tone="warning" title="VRAM headroom is TIGHT">
              Reserved peak {VRAM.rmax.toFixed(1)} GB / {VRAM.capGB.toFixed(1)} GB =
              {" "}{RESERVED_PCT.toFixed(1)}%. Fragmentation only{" "}
              {FRAG_PCT.toFixed(1)}% (excellent) and 0 allocator retires, so the
              run is stable at this shape — but eval/checkpoint save or any seq
              variation will OOM. Activations dominate (~145 GB at seq=8192 with
              no recompute); enabling <Text as="span">selective recompute</Text>{" "}
              would buy ~30-50 GB headroom for ~5-15% throughput cost.
            </Callout>
          </Stack>
        </CardBody>
      </Card>

      <Callout tone="neutral" title="Methodology">
        Trace produced by <Text as="span">torch.profiler</Text> with{" "}
        <Text as="span">record_shapes=false, with_stack=false</Text>; profile
        window steps 80-85, analysis on ProfilerStep#82 (steady-state). Numbers
        from <Text as="span">.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py</Text>;
        opaque RCCL kernels are categorized via the{" "}
        <Text as="span">SPLIT_NCCL_BY_CPU</Text> pass aligning kernel timestamps
        with <Text as="span">c10d::*</Text> CPU ops. VRAM stats from{" "}
        <Text as="span">train_utils.py:671</Text> at iter 10 of the same run.
      </Callout>
    </Stack>
  );
}
