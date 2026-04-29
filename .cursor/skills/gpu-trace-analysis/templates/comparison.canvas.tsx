/**
 * Template: AMD vs NVIDIA trace comparison.
 *
 * Usage from the gpu-trace-analysis skill:
 *   1. Copy this file to ~/.cursor/projects/<workspace>/canvases/<name>.canvas.tsx
 *   2. Replace AMD / NVIDIA constants below with numbers from
 *      scripts/full_breakdown.py for both runs.
 *   3. Update the H1, the Pill strip, the LANES arrays, and the analysis cards.
 *
 * Conventions:
 *   - "amd"/"nv" are the canonical keys for left/right side. Re-label to the
 *     specific SKU (MI355X, B200, …) only in display strings.
 *   - The shipped values match a real run (gpt-oss-20B FP8: MI355X EP=1 vs
 *     B200 EP=8) and should be replaced.
 *   - SEG_COLORS is the only place hex appears; everything else uses
 *     useHostTheme() tokens.
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
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

/* ── Run identity (left = AMD, right = NVIDIA by convention) ─────────── */
const RUN = {
  amd: {
    label: "MI355X",
    config: "TP1 PP1 EP1 · GBS 32 · MBS 4",
    stepMs: 1129.33,
    samples: 32,
    activeStreams: 8,
    computeBusyPct: 95,
    nccLHiddenPct: 81,
    streamOversubPct: 152,
  },
  nv: {
    label: "B200",
    config: "TP1 PP1 EP8 · GBS 16 · MBS 2",
    stepMs: 966.76,
    samples: 16,
    activeStreams: 9,
    computeBusyPct: 43,
    nccLHiddenPct: 20,
    streamOversubPct: 107,
  },
  workload: "gpt-oss-20B FP8 hybrid",
  step: "ProfilerStep #17",
  nodes: "single node · 8 GPUs",
};

/* ── Per-kernel-category busy time across all streams (ms) ─────────────
   Kernel categories come straight from full_breakdown.py output.        */
type Row = { category: string; amd: number; nv: number };
const KERNEL_ROWS: Row[] = [
  { category: "GEMM (dense + grouped)", amd: 940.6, nv: 107.4 },
  { category: "Attention (FMHA fwd/bwd)", amd: 203.1, nv: 64.3 },
  { category: "RMSNorm / LayerNorm", amd: 38.6, nv: 30.0 },
  { category: "MoE dispatch / permute / topk", amd: 27.7, nv: 33.4 },
  { category: "Elementwise / cast / activation", amd: 105.4, nv: 69.0 },
  { category: "Optimizer / param update", amd: 17.4, nv: 11.2 },
  { category: "MemCopy / alloc", amd: 8.5, nv: 4.1 },
  { category: "NCCL AllToAll (MoE expert)", amd: 0.0, nv: 284.4 },
  { category: "NCCL ReduceScatter (DDP)", amd: 96.0, nv: 193.1 },
  { category: "NCCL AllGather (DDP)", amd: 81.0, nv: 162.0 },
  { category: "NCCL AllReduce", amd: 65.0, nv: 19.6 },
];

const STREAM_ROWS_AMD = [
  { stream: "stream 0", role: "Compute (attn / GEMM / norm / opt)", busyMs: 1075, share: "95.2%" },
  { stream: "stream 11", role: "RCCL ReduceScatter / AllGather / AllReduce", busyMs: 240, share: "21.3%" },
  { stream: "stream 13", role: "Parallel grouped GEMM (expert lane)", busyMs: 222, share: "19.7%" },
  { stream: "stream 14", role: "Parallel grouped GEMM (expert lane)", busyMs: 199, share: "17.6%" },
  { stream: "stream 15", role: "Parallel grouped GEMM (expert lane)", busyMs: 192, share: "17.0%" },
  { stream: "stream 16", role: "Parallel grouped GEMM (expert lane)", busyMs: 184, share: "16.3%" },
];
const STREAM_ROWS_NV = [
  { stream: "stream 7", role: "Compute + MoE all-to-all (mixed)", busyMs: 617, share: "63.8%" },
  { stream: "stream 55", role: "NCCL ReduceScatter / AllGather", busyMs: 257, share: "26.6%" },
  { stream: "stream 51", role: "NCCL ReduceScatter / AllGather", busyMs: 80, share: "8.3%" },
  { stream: "stream 173", role: "Aux GEMM tail", busyMs: 35, share: "3.6%" },
  { stream: "stream 174", role: "Aux GEMM tail", busyMs: 20, share: "2.1%" },
];

/* ── Multi-stream pipeline lanes (one ProfilerStep, normalized to 100) ── */
type SegKind =
  | "gemm" | "attn" | "norm" | "moe" | "elem" | "opt"
  | "a2a" | "rs" | "ag" | "ar" | "idle";
type Seg = { kind: SegKind; t: number; w: number; label?: string };

const AMD_LANES: Array<{ name: string; tag: string; segs: Seg[] }> = [
  {
    name: "stream 0", tag: "compute",
    segs: [
      { kind: "elem", t: 0, w: 2 },
      { kind: "norm", t: 2, w: 2 },
      { kind: "gemm", t: 4, w: 6, label: "qkv proj" },
      { kind: "attn", t: 10, w: 10, label: "fmha fwd" },
      { kind: "gemm", t: 20, w: 4 },
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
    name: "stream 13", tag: "expert lane A",
    segs: [
      { kind: "idle", t: 0, w: 30 },
      { kind: "gemm", t: 30, w: 18, label: "grouped GEMM" },
      { kind: "idle", t: 48, w: 18 },
      { kind: "gemm", t: 66, w: 22, label: "grouped GEMM bwd" },
      { kind: "idle", t: 88, w: 12 },
    ],
  },
  {
    name: "stream 14", tag: "expert lane B",
    segs: [
      { kind: "idle", t: 0, w: 30 },
      { kind: "gemm", t: 30, w: 17 },
      { kind: "idle", t: 47, w: 19 },
      { kind: "gemm", t: 66, w: 21 },
      { kind: "idle", t: 87, w: 13 },
    ],
  },
  {
    name: "stream 11", tag: "RCCL DDP",
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

const NV_LANES: Array<{ name: string; tag: string; segs: Seg[] }> = [
  {
    name: "stream 7", tag: "compute + a2a",
    segs: [
      { kind: "elem", t: 0, w: 2 },
      { kind: "norm", t: 2, w: 2 },
      { kind: "gemm", t: 4, w: 4, label: "fwd qkv" },
      { kind: "attn", t: 8, w: 4, label: "fmha fwd" },
      { kind: "gemm", t: 12, w: 3 },
      { kind: "moe", t: 15, w: 3, label: "topk+permute" },
      { kind: "a2a", t: 18, w: 14, label: "sendrecv (dispatch)" },
      { kind: "gemm", t: 32, w: 6, label: "expert MLP" },
      { kind: "a2a", t: 38, w: 14, label: "sendrecv (combine)" },
      { kind: "elem", t: 52, w: 2 },
      { kind: "norm", t: 54, w: 2 },
      { kind: "attn", t: 56, w: 6, label: "fmha bwd" },
      { kind: "gemm", t: 62, w: 4 },
      { kind: "moe", t: 66, w: 3 },
      { kind: "a2a", t: 69, w: 12, label: "sendrecv (bwd)" },
      { kind: "gemm", t: 81, w: 6 },
      { kind: "ar", t: 87, w: 2 },
      { kind: "elem", t: 89, w: 3 },
      { kind: "opt", t: 92, w: 4, label: "step" },
      { kind: "idle", t: 96, w: 4 },
    ],
  },
  {
    name: "stream 55", tag: "NCCL DDP",
    segs: [
      { kind: "idle", t: 0, w: 30 },
      { kind: "rs", t: 30, w: 14, label: "reduce-scatter" },
      { kind: "idle", t: 44, w: 8 },
      { kind: "ag", t: 52, w: 12, label: "all-gather" },
      { kind: "idle", t: 64, w: 36 },
    ],
  },
  {
    name: "stream 51", tag: "NCCL DDP",
    segs: [
      { kind: "idle", t: 0, w: 70 },
      { kind: "rs", t: 70, w: 6 },
      { kind: "ag", t: 76, w: 2 },
      { kind: "idle", t: 78, w: 22 },
    ],
  },
  {
    name: "stream 173", tag: "aux GEMM",
    segs: [
      { kind: "idle", t: 0, w: 40 },
      { kind: "gemm", t: 40, w: 4 },
      { kind: "idle", t: 44, w: 50 },
      { kind: "gemm", t: 94, w: 2 },
      { kind: "idle", t: 96, w: 4 },
    ],
  },
];

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
  title,
  totalMs,
  lanes,
}: {
  title: string;
  totalMs: number;
  lanes: Array<{ name: string; tag: string; segs: Seg[] }>;
}) {
  const t = useHostTheme();
  const palette = useSegPalette();
  const labelW = 150;
  const trackH = 22;
  const gap = 6;
  const width = 720;
  const trackW = width - labelW;
  const height = lanes.length * (trackH + gap) + 28;

  return (
    <Stack gap={8}>
      <Row align="center" gap={8}>
        <Text weight="semibold">{title}</Text>
        <Text tone="secondary" size="small">
          one ProfilerStep ≈ {totalMs.toFixed(0)} ms · time →
        </Text>
      </Row>
      <svg width={width} height={height} style={{ display: "block" }}>
        {[0, 25, 50, 75, 100].map((p) => {
          const x = labelW + (trackW * p) / 100;
          return (
            <g key={p}>
              <line x1={x} x2={x} y1={0}
                y2={lanes.length * (trackH + gap)}
                stroke={t.stroke.tertiary} strokeDasharray="2 3" />
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
    </Stack>
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

const fmt = (n: number, d = 1) => n.toFixed(d);

export default function AmdVsNvidiaTrace() {
  const msPerSampleAmd = RUN.amd.stepMs / RUN.amd.samples;
  const msPerSampleNv = RUN.nv.stepMs / RUN.nv.samples;
  const fasterPerSample = msPerSampleAmd < msPerSampleNv ? "AMD" : "NVIDIA";
  const speedup =
    fasterPerSample === "AMD"
      ? msPerSampleNv / msPerSampleAmd
      : msPerSampleAmd / msPerSampleNv;

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>{RUN.amd.label} vs {RUN.nv.label} — {RUN.workload} trace breakdown</H1>
        <Row gap={8} wrap>
          <Pill tone="info">{RUN.nodes} · {RUN.step}</Pill>
          <Pill tone="neutral">{RUN.amd.label}: {RUN.amd.config}</Pill>
          <Pill tone="neutral">{RUN.nv.label}: {RUN.nv.config}</Pill>
        </Row>
      </Stack>

      <Callout tone="success" title="Headline">
        Per-sample {fasterPerSample} delivers{" "}
        <Text as="span" weight="semibold">
          {fmt(Math.min(msPerSampleAmd, msPerSampleNv))} ms/sample
        </Text>{" "}
        vs{" "}
        <Text as="span" weight="semibold">
          {fmt(Math.max(msPerSampleAmd, msPerSampleNv))} ms/sample
        </Text>{" "}
        — a {fmt(speedup, 2)}× per-token throughput advantage in this
        configuration. Replace this paragraph with the actual cause (e.g.
        memory-driven EP choice, communication on critical path, kernel-level
        TFLOP/s differences).
      </Callout>

      <Grid columns={5} gap={12}>
        <Stat value={`${fmt(RUN.amd.stepMs, 0)} ms`} label={`${RUN.amd.label} step time`} />
        <Stat value={`${fmt(RUN.nv.stepMs, 0)} ms`} label={`${RUN.nv.label} step time`} />
        <Stat value={`${fmt(speedup, 2)}×`}
          label={`${fasterPerSample} per-sample speed-up`} tone="success" />
        <Stat value={`${RUN.nv.computeBusyPct}%`}
          label={`${RUN.nv.label} compute-stream busy`}
          tone={RUN.nv.computeBusyPct > 80 ? "success" : "warning"} />
        <Stat value={`${RUN.amd.computeBusyPct}%`}
          label={`${RUN.amd.label} compute-stream busy`}
          tone={RUN.amd.computeBusyPct > 80 ? "success" : "warning"} />
      </Grid>

      <Divider />

      <H2>Per-stream timeline (one ProfilerStep)</H2>
      <Text tone="secondary">
        Each lane is a HIP/CUDA stream. Width is normalized to the step
        duration; segments are colored by kernel category.
      </Text>
      <Legend />

      <Card>
        <CardHeader trailing={
          <Pill tone={RUN.nv.computeBusyPct < 60 ? "warning" : "success"}>
            {RUN.nv.computeBusyPct < 60 ? "communication-bound" : "compute-bound"}
          </Pill>
        }>
          {RUN.nv.label} · {RUN.nv.config}
        </CardHeader>
        <CardBody>
          <PipelineDiagram
            title="NVIDIA — replace title with the structural finding."
            totalMs={RUN.nv.stepMs}
            lanes={NV_LANES}
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={
          <Pill tone={RUN.amd.computeBusyPct < 60 ? "warning" : "success"}>
            {RUN.amd.computeBusyPct < 60 ? "communication-bound" : "compute-bound"}
          </Pill>
        }>
          {RUN.amd.label} · {RUN.amd.config}
        </CardHeader>
        <CardBody>
          <PipelineDiagram
            title="AMD — replace title with the structural finding."
            totalMs={RUN.amd.stepMs}
            lanes={AMD_LANES}
          />
        </CardBody>
      </Card>

      <Divider />

      <H2>GPU work decomposition</H2>
      <Text tone="secondary">
        Sum of busy time per kernel category across all streams (ms). Numbers
        can exceed step time when multiple streams run in parallel.
      </Text>

      <Grid columns={2} gap={16}>
        <Stack gap={8}>
          <H3>Kernel breakdown (ms)</H3>
          <Table
            headers={["Category", RUN.amd.label, RUN.nv.label, "Δ"]}
            columnAlign={["left", "right", "right", "right"]}
            rows={KERNEL_ROWS.map((r) => [
              r.category, fmt(r.amd), fmt(r.nv), fmt(r.amd - r.nv),
            ])}
            rowTone={KERNEL_ROWS.map((r) => {
              if (r.category.startsWith("NCCL AllToAll")) return "warning";
              if (r.category.startsWith("GEMM")) return "info";
              return undefined;
            })}
          />
        </Stack>
        <Stack gap={8}>
          <H3>Stacked share by category</H3>
          <BarChart
            categories={[RUN.amd.label, RUN.nv.label]}
            stacked normalized height={360} valueSuffix="%"
            series={[
              { name: "GEMM", tone: "info",
                data: pickPair("GEMM") },
              { name: "Attention",
                data: pickPair("Attention") },
              { name: "Norm/Elem/Opt/Mem", tone: "neutral",
                data: sumPair(["RMSNorm", "Elementwise", "Optimizer", "MemCopy"]) },
              { name: "MoE dispatch",
                data: pickPair("MoE") },
              { name: "NCCL all-to-all", tone: "danger",
                data: pickPair("NCCL AllToAll") },
              { name: "NCCL DDP (RS+AG+AR)", tone: "warning",
                data: sumPair(["NCCL ReduceScatter", "NCCL AllGather", "NCCL AllReduce"]) },
            ]}
          />
        </Stack>
      </Grid>

      <Divider />

      <H2>Stream occupancy</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader trailing={<Pill size="sm">{RUN.amd.activeStreams} active streams</Pill>}>
            {RUN.amd.label} — per-stream busy time
          </CardHeader>
          <CardBody>
            <Table
              headers={["Stream", "Role", "Busy (ms)", "Share of step"]}
              columnAlign={["left", "left", "right", "right"]}
              rows={STREAM_ROWS_AMD.map((r) => [
                r.stream, r.role, fmt(r.busyMs, 0), r.share,
              ])}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">{RUN.nv.activeStreams} active streams</Pill>}>
            {RUN.nv.label} — per-stream busy time
          </CardHeader>
          <CardBody>
            <Table
              headers={["Stream", "Role", "Busy (ms)", "Share of step"]}
              columnAlign={["left", "left", "right", "right"]}
              rows={STREAM_ROWS_NV.map((r) => [
                r.stream, r.role, fmt(r.busyMs, 0), r.share,
              ])}
            />
          </CardBody>
        </Card>
      </Grid>

      <Grid columns={4} gap={12}>
        <Stat value={`${RUN.nv.nccLHiddenPct}%`}
          label={`${RUN.nv.label} NCCL hidden`}
          tone={RUN.nv.nccLHiddenPct > 60 ? "success" : "danger"} />
        <Stat value={`${RUN.amd.nccLHiddenPct}%`}
          label={`${RUN.amd.label} NCCL hidden`}
          tone={RUN.amd.nccLHiddenPct > 60 ? "success" : "danger"} />
        <Stat value={`${RUN.nv.streamOversubPct}%`}
          label={`${RUN.nv.label} stream oversubscription`} />
        <Stat value={`${RUN.amd.streamOversubPct}%`}
          label={`${RUN.amd.label} stream oversubscription`}
          tone={RUN.amd.streamOversubPct > 120 ? "success" : undefined} />
      </Grid>

      <Divider />

      <H2>What the timelines tell us</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>{RUN.nv.label} — replace with finding</CardHeader>
          <CardBody>
            <Text>
              Describe the critical-path operation, the streams it serializes
              with, and the headroom (e.g. moving MoE a2a to its own NCCL
              stream so it overlaps the next micro-batch&apos;s attention).
            </Text>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>{RUN.amd.label} — replace with finding</CardHeader>
          <CardBody>
            <Text>
              Describe what dominates the compute stream and where the
              remaining headroom is (kernel-internal vs scheduling).
            </Text>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="info" title="Methodology">
        Traces produced by torch.profiler with{" "}
        <Text as="span">record_shapes=false, with_stack=false</Text>; one
        active step per rank. Numbers come from the same{" "}
        <Text as="span">{RUN.step}</Text> on a non-rank-0 rank for both
        sides. Opaque RCCL kernels are categorized via{" "}
        <Text as="span">SPLIT_NCCL_BY_CPU</Text> in the breakdown script.
      </Callout>
    </Stack>
  );
}

/* ── helpers used by the BarChart series ─────────────────────────────── */
function pickPair(prefix: string): [number, number] {
  const r = KERNEL_ROWS.find((x) => x.category.startsWith(prefix));
  return r ? [r.amd, r.nv] : [0, 0];
}
function sumPair(prefixes: string[]): [number, number] {
  let a = 0, n = 0;
  for (const p of prefixes) {
    const r = KERNEL_ROWS.find((x) => x.category.startsWith(p));
    if (r) { a += r.amd; n += r.nv; }
  }
  return [a, n];
}
