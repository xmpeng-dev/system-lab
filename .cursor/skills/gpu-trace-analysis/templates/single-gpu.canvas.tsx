/**
 * Template: single-GPU trace deep-dive (e.g. an AMD MI355X run).
 *
 * Usage from the gpu-trace-analysis skill:
 *   1. Copy this file to ~/.cursor/projects/<workspace>/canvases/<name>.canvas.tsx
 *   2. Replace the data constants near the top with numbers from
 *      scripts/full_breakdown.py.
 *   3. Update the H1, the Pill strip, the LANES array, and the analysis card.
 *
 * Rules:
 *   - Only import from "cursor/canvas".
 *   - SEG_COLORS is the single source of hex; everything else uses theme tokens.
 */
import {
  BarChart,
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
  hardware: "8 × MI355X",
  model: "gpt-oss-20B FP8 hybrid",
  parallelism: "TP1 PP1 EP1",
  batch: "GBS 32 · MBS 4",
  step: "ProfilerStep #17",
  stepMs: 1129.33,
  samplesPerStep: 32,
};

/* ── Per-kernel-category busy time across all streams (ms) ────────────── */
type KernelRow = { category: string; ms: number; tone?: "info" | "warning" };
const KERNEL_ROWS: KernelRow[] = [
  { category: "GEMM (dense + grouped)", ms: 940.6, tone: "info" },
  { category: "Attention (FMHA fwd/bwd)", ms: 203.1 },
  { category: "Elementwise / cast / activation", ms: 105.4 },
  { category: "NCCL ReduceScatter (DDP)", ms: 96.0, tone: "warning" },
  { category: "NCCL AllGather (DDP)", ms: 81.0, tone: "warning" },
  { category: "NCCL AllReduce", ms: 65.0, tone: "warning" },
  { category: "RMSNorm / LayerNorm", ms: 38.6 },
  { category: "MoE dispatch / permute / topk", ms: 27.7 },
  { category: "Optimizer / param update", ms: 17.4 },
  { category: "MemCopy / alloc", ms: 8.5 },
];

const STREAM_ROWS: Array<{
  stream: string;
  role: string;
  busyMs: number;
  share: string;
}> = [
  { stream: "stream 0", role: "Compute (attn / GEMM / norm / opt)", busyMs: 1075, share: "95.2%" },
  { stream: "stream 11", role: "RCCL ReduceScatter / AllGather / AllReduce", busyMs: 240, share: "21.3%" },
  { stream: "stream 13", role: "Parallel grouped GEMM (expert lane)", busyMs: 222, share: "19.7%" },
  { stream: "stream 14", role: "Parallel grouped GEMM (expert lane)", busyMs: 199, share: "17.6%" },
  { stream: "stream 15", role: "Parallel grouped GEMM (expert lane)", busyMs: 192, share: "17.0%" },
  { stream: "stream 16", role: "Parallel grouped GEMM (expert lane)", busyMs: 184, share: "16.3%" },
  { stream: "others (2)", role: "MemCopy / D2D", busyMs: 12, share: "1.1%" },
];

/* ── Multi-stream pipeline lanes for one ProfilerStep ─────────────────── */
type SegKind =
  | "gemm" | "attn" | "norm" | "moe" | "elem" | "opt"
  | "a2a" | "rs" | "ag" | "ar" | "idle";
type Seg = { kind: SegKind; t: number; w: number; label?: string };

const LANES: Array<{ name: string; tag: string; segs: Seg[] }> = [
  {
    name: "stream 0",
    tag: "compute",
    segs: [
      { kind: "elem", t: 0, w: 2 },
      { kind: "norm", t: 2, w: 2 },
      { kind: "gemm", t: 4, w: 6, label: "qkv proj" },
      { kind: "attn", t: 10, w: 10, label: "aiter fmha fwd" },
      { kind: "gemm", t: 20, w: 4, label: "out proj" },
      { kind: "norm", t: 24, w: 2 },
      { kind: "moe", t: 26, w: 3 },
      { kind: "gemm", t: 29, w: 18, label: "grouped MLP fwd" },
      { kind: "moe", t: 47, w: 2 },
      { kind: "elem", t: 49, w: 3 },
      { kind: "norm", t: 52, w: 2 },
      { kind: "attn", t: 54, w: 12, label: "fmha bwd" },
      { kind: "gemm", t: 66, w: 22, label: "grouped MLP bwd" },
      { kind: "elem", t: 88, w: 4 },
      { kind: "opt", t: 92, w: 6, label: "optim step" },
      { kind: "idle", t: 98, w: 2 },
    ],
  },
  {
    name: "stream 13",
    tag: "expert lane A",
    segs: [
      { kind: "idle", t: 0, w: 30 },
      { kind: "gemm", t: 30, w: 18, label: "grouped GEMM" },
      { kind: "idle", t: 48, w: 18 },
      { kind: "gemm", t: 66, w: 22, label: "grouped GEMM bwd" },
      { kind: "idle", t: 88, w: 12 },
    ],
  },
  {
    name: "stream 14",
    tag: "expert lane B",
    segs: [
      { kind: "idle", t: 0, w: 30 },
      { kind: "gemm", t: 30, w: 17 },
      { kind: "idle", t: 47, w: 19 },
      { kind: "gemm", t: 66, w: 21 },
      { kind: "idle", t: 87, w: 13 },
    ],
  },
  {
    name: "stream 15",
    tag: "expert lane C",
    segs: [
      { kind: "idle", t: 0, w: 30 },
      { kind: "gemm", t: 30, w: 17 },
      { kind: "idle", t: 47, w: 19 },
      { kind: "gemm", t: 66, w: 20 },
      { kind: "idle", t: 86, w: 14 },
    ],
  },
  {
    name: "stream 11",
    tag: "RCCL DDP",
    segs: [
      { kind: "idle", t: 0, w: 26 },
      { kind: "rs", t: 26, w: 6, label: "RS" },
      { kind: "idle", t: 32, w: 18 },
      { kind: "ar", t: 50, w: 4 },
      { kind: "idle", t: 54, w: 30 },
      { kind: "ag", t: 84, w: 6, label: "AG" },
      { kind: "rs", t: 90, w: 4 },
      { kind: "idle", t: 94, w: 6 },
    ],
  },
];

const COMPUTE_BUSY_PCT = 95.2;
const NCCL_HIDDEN_PCT = 81.0;
const STREAM_OVERSUB_PCT = 152;

/* ── Palette aligned with BarChart series colors ──────────────────────── */
const SEG_COLORS: Record<SegKind, string> = {
  gemm: "#2E79B5E0",
  attn: "#7B64B8F0",
  norm: "#70B0D8E0",
  moe: "#F0A040E0",
  elem: "#8888A8E0",
  opt: "#1F8A65E8",
  a2a: "#C04848E0",
  rs: "#E8C030E0",
  ag: "#7DCAB0E0",
  ar: "#C85898E0",
  idle: "transparent",
};

function useSegPalette() {
  const t = useHostTheme();
  return { ...SEG_COLORS, idle: t.fill.tertiary } as Record<SegKind, string>;
}

function PipelineDiagram({
  totalMs,
  lanes,
  width = 720,
}: {
  totalMs: number;
  lanes: Array<{ name: string; tag: string; segs: Seg[] }>;
  width?: number;
}) {
  const t = useHostTheme();
  const palette = useSegPalette();
  const labelW = 150;
  const trackH = 22;
  const gap = 6;
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
            <text x={70} y={y + trackH / 2 + 4} fontSize={10} fill={t.text.quaternary}>
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
                    <title>{`${s.kind}${s.label ? " · " + s.label : ""} (${s.w}%)`}</title>
                  </rect>
                  {s.label && sw > 38 ? (
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
    { kind: "gemm", label: "GEMM" },
    { kind: "attn", label: "Attention" },
    { kind: "norm", label: "Norm" },
    { kind: "moe", label: "MoE dispatch" },
    { kind: "elem", label: "Elementwise" },
    { kind: "opt", label: "Optimizer" },
    { kind: "a2a", label: "NCCL all-to-all" },
    { kind: "rs", label: "ReduceScatter" },
    { kind: "ag", label: "AllGather" },
    { kind: "ar", label: "AllReduce" },
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

export default function SingleGpuTrace() {
  const msPerSample = RUN.stepMs / RUN.samplesPerStep;

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>{RUN.hardware} — {RUN.model} trace breakdown</H1>
        <Row gap={8} wrap>
          <Pill tone="info">{RUN.step}</Pill>
          <Pill tone="neutral">{RUN.parallelism}</Pill>
          <Pill tone="neutral">{RUN.batch}</Pill>
        </Row>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat value={`${RUN.stepMs.toFixed(0)} ms`} label="step time" />
        <Stat value={`${msPerSample.toFixed(1)} ms`} label="per sample" />
        <Stat value={`${COMPUTE_BUSY_PCT.toFixed(0)}%`}
          label="compute-stream busy"
          tone={COMPUTE_BUSY_PCT > 80 ? "success" : "warning"} />
        <Stat value={`${NCCL_HIDDEN_PCT.toFixed(0)}%`}
          label="NCCL hidden behind compute"
          tone={NCCL_HIDDEN_PCT > 60 ? "success" : "warning"} />
      </Grid>

      <Divider />

      <H2>Per-stream timeline (one ProfilerStep)</H2>
      <Text tone="secondary">
        Each lane is a HIP/CUDA stream. Width is normalized to the step
        duration; segments are colored by kernel category.
      </Text>
      <Legend />
      <PipelineDiagram totalMs={RUN.stepMs} lanes={LANES} />

      <Divider />

      <H2>GPU work decomposition</H2>
      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <Text tone="secondary">Sum of busy time per kernel category across all streams (ms).</Text>
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
        </Stack>
        <Stack gap={8}>
          <Text tone="secondary">Top streams by busy time.</Text>
          <Table
            headers={["Stream", "Role", "ms", "Share"]}
            columnAlign={["left", "left", "right", "right"]}
            rows={STREAM_ROWS.map((r) => [
              r.stream, r.role, r.busyMs.toFixed(0), r.share,
            ])}
          />
        </Stack>
      </Grid>

      <Card>
        <CardHeader trailing={<Pill size="sm">{STREAM_OVERSUB_PCT}% oversub</Pill>}>
          What the timeline tells us
        </CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text>
              Stream 0 carries the dense compute path; expert grouped-GEMMs are
              dispatched onto streams 13–16 and run concurrently, pushing the
              total GEMM work to{" "}
              <Text as="span" weight="semibold">
                {KERNEL_ROWS.find((r) => r.category.startsWith("GEMM"))!.ms.toFixed(0)} ms
              </Text>{" "}
              inside a {RUN.stepMs.toFixed(0)} ms step.
            </Text>
            <Text>
              RCCL DDP traffic on stream 11 (RS / AG / AR) is{" "}
              {NCCL_HIDDEN_PCT.toFixed(0)}% hidden behind compute, so collective
              cost is essentially free.
            </Text>
            <Callout tone="info" title="Replace this paragraph with the actual finding">
              e.g. "compute-bound, headroom is inside FMHA bwd kernel" or
              "exposed bubble around optimizer step / weight broadcast".
            </Callout>
          </Stack>
        </CardBody>
      </Card>

      <Callout tone="neutral" title="Methodology">
        Trace produced by torch.profiler with{" "}
        <Text as="span">record_shapes=false, with_stack=false</Text>; one
        active step per rank. Numbers come from a non-rank-0 rank (other ranks
        are within ±3%). Opaque RCCL kernels are categorized via the{" "}
        <Text as="span">SPLIT_NCCL_BY_CPU</Text> pass that aligns kernel
        timestamps with <Text as="span">c10d::*</Text> CPU ops.
      </Callout>
    </Stack>
  );
}
