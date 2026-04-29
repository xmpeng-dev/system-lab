# cursor/canvas → HTML mapping reference

Companion to `SKILL.md` and `template.html`. Look up each `cursor/canvas`
component you encounter in the source `.canvas.tsx` and emit the
corresponding HTML pattern. All class names are pre-defined in
`template.html`.

## Theme tokens

When the source canvas reads `useHostTheme()` (typical inside hand-written
`<svg>` blocks), drop the hook call and replace each token access with the
matching CSS variable. Both light and dark mappings are pre-wired in
`template.html` via `prefers-color-scheme`.

| `useHostTheme()` access | CSS variable |
|---|---|
| `t.text.primary` | `var(--text-primary)` |
| `t.text.secondary` | `var(--text-secondary)` |
| `t.text.tertiary` | `var(--text-tertiary)` |
| `t.text.quaternary` | `var(--text-quaternary)` |
| `t.text.onAccent` | `var(--text-on-accent)` |
| `t.fill.primary` | `var(--fill-primary)` |
| `t.fill.secondary` | `var(--fill-secondary)` |
| `t.fill.tertiary` | `var(--fill-tertiary)` |
| `t.fill.quaternary` | `var(--fill-quaternary)` |
| `t.stroke.primary` | `var(--stroke-primary)` |
| `t.stroke.secondary` | `var(--stroke-secondary)` |
| `t.stroke.tertiary` | `var(--stroke-tertiary)` |
| accent / success / warning / danger / info | `var(--accent)` etc. |

## Layout

### `Stack`
```tsx
<Stack gap={20}>…</Stack>
```
```html
<div class="stack" style="gap:20px">…</div>
```

### `Row`
```tsx
<Row gap={8} align="center" wrap>…</Row>
```
```html
<div class="row" style="gap:8px">…</div>
```
- `align="start"` → add `style="align-items:flex-start"` (or class `row-top`).
- `justify="space-between"` → add class `row-between`.

### `Grid`
```tsx
<Grid columns={3} gap={16}>…</Grid>
```
```html
<div class="grid grid-3" style="gap:16px">…</div>
```
- Pre-defined column classes: `grid-2` … `grid-6`.
- For non-uniform columns (`columns="1fr 2fr"`), use inline style:
  `<div class="grid" style="grid-template-columns:1fr 2fr;gap:16px">`.

### `Divider`
```tsx
<Divider />
```
```html
<hr class="divider">
```

### `Spacer`
Inside a `Row`, replace with a flex spacer:
```html
<span style="flex:1"></span>
```

## Typography

### `H1` / `H2` / `H3`
```html
<h1>Title</h1>
<h2>Section</h2>
<h3>Subsection</h3>
```

### `Text`
```tsx
<Text tone="secondary" size="small" weight="semibold">…</Text>
```
```html
<p class="text text-secondary text-small text-semibold">…</p>
```
- Tone classes: `text-secondary`, `text-tertiary`, `text-quaternary`.
- `size="small"` → `text-small`.
- `weight="semibold"` / `weight="bold"` → `text-semibold` (use `<strong>`
  for bold inline emphasis).
- `italic` → `text-italic`.
- When the source uses `<Text as="span">` inline inside another text, emit
  `<span>…</span>` instead of `<p>`.
- For monospaced numbers, add `text-num` (already enables `font-feature-settings: "tnum"`).

### `Code`
```tsx
<Code>npm install</Code>
```
```html
<code>npm install</code>
```

### `Link`
```tsx
<Link href="https://x">docs</Link>
```
```html
<a href="https://x" target="_blank" rel="noopener">docs</a>
```

## Card

```tsx
<Card>
  <CardHeader trailing={<Pill tone="success">healthy</Pill>}>
    file.ts
  </CardHeader>
  <CardBody>…body…</CardBody>
</Card>
```
```html
<div class="card">
  <div class="card-header">
    <span>file.ts</span>
    <span class="pill pill-success">healthy</span>
  </div>
  <div class="card-body">…body…</div>
</div>
```
- If `CardBody` has `style={{ padding: 0 }}`, swap class to
  `card-body-flush`.
- `Card collapsible` is rare; render expanded by default and skip the
  toggle (no JS).

## Stat

```tsx
<Stat value="42%" label="utilization" tone="success" />
```
```html
<div class="stat stat-success">
  <div class="stat-value">42%</div>
  <div class="stat-label">utilization</div>
</div>
```
- Tone classes: `stat-success`, `stat-warning`, `stat-danger`, `stat-info`.
- Omit the tone class for default neutral.

## Pill

```tsx
<Pill tone="info">EP=8</Pill>
<Pill size="sm">9 streams</Pill>
```
```html
<span class="pill pill-info">EP=8</span>
<span class="pill pill-sm">9 streams</span>
```
- Tone classes: `pill-success`, `pill-warning`, `pill-danger`, `pill-info`,
  `pill-neutral`.
- `size="sm"` → add `pill-sm` (overrides border, smaller padding).

## Callout

```tsx
<Callout tone="warning" title="Heads up">
  Rolling deploy in progress.
</Callout>
```
```html
<div class="callout callout-warning">
  <div class="callout-title">Heads up</div>
  <p>Rolling deploy in progress.</p>
</div>
```
- Tone classes: `callout-success`, `callout-warning`, `callout-danger`,
  `callout-info`. Default is neutral (no extra class).

## Table

```tsx
<Table
  headers={["Service", "Status", "RPS"]}
  columnAlign={["left", "left", "right"]}
  rows={[
    ["api", "ok", "3.2k"],
    ["worker", "warn", "8.1k"],
  ]}
  rowTone={[undefined, "warning"]}
/>
```
```html
<div class="table-shell">
  <table class="table">
    <thead>
      <tr>
        <th>Service</th>
        <th>Status</th>
        <th class="right">RPS</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>api</td><td>ok</td><td class="right">3.2k</td></tr>
      <tr class="row-warning"><td>worker</td><td>warn</td><td class="right">8.1k</td></tr>
    </tbody>
  </table>
</div>
```
- Per-column alignment: add `class="right"` or `class="center"` to BOTH
  the `<th>` and every matching `<td>` in that column.
- Row tones: `row-success`, `row-warning`, `row-danger`, `row-info`,
  `row-neutral`. Sparse `undefined` rows get no class.
- If `framed={false}` on the source `<Table>`, drop the wrapping
  `<div class="table-shell">`.
- For `striped`, add a CSS rule once at top of `<style>`:
  `.table tr:nth-child(even) td { background: var(--fill-quaternary); }`.

## Custom inline `<svg>` blocks

Many gpu-trace canvases hand-roll an `<svg>` (pipeline diagram, lane chart).
Copy the SVG **verbatim** into the HTML. Then sweep through and:

1. Replace every `useHostTheme()` token with the matching `var(--…)` from
   the table at the top of this file.
2. JSX uses camelCase attributes (`strokeDasharray`, `textAnchor`,
   `fontFamily`, `fontSize`, `pointerEvents`); convert to kebab-case for
   HTML (`stroke-dasharray`, `text-anchor`, `font-family`, `font-size`,
   `pointer-events`). Conversion table for the most common ones:

   | JSX | HTML |
   |---|---|
   | `strokeDasharray` | `stroke-dasharray` |
   | `strokeWidth` | `stroke-width` |
   | `strokeLinecap` | `stroke-linecap` |
   | `textAnchor` | `text-anchor` |
   | `dominantBaseline` | `dominant-baseline` |
   | `fontSize` | `font-size` |
   | `fontFamily` | `font-family` |
   | `fontWeight` | `font-weight` |
   | `xlinkHref` | `xlink:href` |
   | `clipPath` | `clip-path` |
   | `pointerEvents` | `pointer-events` |

3. Convert React `style={{ key: val }}` objects to inline CSS strings:
   `style="key: val"` (camelCase → kebab-case for property names).
4. Drop React-only attributes: `key`, fragment shorthands, `className`
   becomes `class`.
5. If the SVG references a `palette[s.kind]` object built from
   `SEG_COLORS`, inline the resolved hex/oklch into each `fill="…"`
   attribute — do not try to recreate the JS lookup at runtime.

## Charts

`cursor/canvas` ships `BarChart`, `LineChart`, `PieChart`. Render each as
inline SVG with the data baked into element attributes (no JS at runtime).
Pick the recipe matching the source props.

### BarChart — single series, vertical

```tsx
<BarChart
  categories={["Mon","Tue","Wed"]}
  series={[{ name: "Requests", data: [120, 90, 150] }]}
  height={200}
/>
```

Pattern (manually pre-compute bar geometry and embed):

```html
<div class="chart-frame">
  <svg viewBox="0 0 400 220" width="100%" height="220"
       font-family="-apple-system, system-ui, sans-serif">
    <!-- y-axis grid + ticks -->
    <g stroke="var(--stroke-tertiary)">
      <line x1="40" x2="400" y1="180" y2="180"/>
      <line x1="40" x2="400" y1="120" y2="120" stroke-dasharray="2 3"/>
      <line x1="40" x2="400" y1="60"  y2="60"  stroke-dasharray="2 3"/>
    </g>
    <g font-size="10" fill="var(--text-tertiary)" text-anchor="end">
      <text x="36" y="184">0</text>
      <text x="36" y="124">75</text>
      <text x="36" y="64">150</text>
    </g>
    <!-- bars -->
    <g fill="var(--chart-1)">
      <rect x="80"  y="84"  width="60" height="96"/>  <!-- 120 -->
      <rect x="180" y="108" width="60" height="72"/>  <!-- 90  -->
      <rect x="280" y="60"  width="60" height="120"/> <!-- 150 -->
    </g>
    <!-- category labels -->
    <g font-size="11" fill="var(--text-secondary)" text-anchor="middle">
      <text x="110" y="200">Mon</text>
      <text x="210" y="200">Tue</text>
      <text x="310" y="200">Wed</text>
    </g>
  </svg>
</div>
```

### BarChart — stacked normalized (100% share)

This is the most common pattern in gpu-trace canvases (`stacked normalized
valueSuffix="%"`). Pre-compute the percentage of each series per category
and stack them vertically.

```tsx
<BarChart
  categories={["A","B"]}
  stacked normalized height={300}
  series={[
    { name: "GEMM", tone: "info",    data: [940, 107] },
    { name: "NCCL", tone: "warning", data: [242, 759] },
  ]}
/>
```

Pre-compute (do this in your head / on paper):
- A total = 1182, GEMM share = 79.5%, NCCL share = 20.5%
- B total = 866,  GEMM share = 12.4%, NCCL share = 87.6%

Pattern (one `<rect>` per stacked segment, 100%-tall column = 240 px):

```html
<div class="chart-frame">
  <svg viewBox="0 0 360 320" width="100%" height="320"
       font-family="-apple-system, system-ui, sans-serif">
    <!-- column A at x=80, width 80, full height 240, y_top=40 -->
    <g>
      <rect x="80" y="40"  width="80" height="190.8" fill="var(--info)"/>     <!-- 79.5% -->
      <rect x="80" y="230.8" width="80" height="49.2" fill="var(--warning)"/> <!-- 20.5% -->
    </g>
    <!-- column B at x=200 -->
    <g>
      <rect x="200" y="40"   width="80" height="29.8"  fill="var(--info)"/>    <!-- 12.4% -->
      <rect x="200" y="69.8" width="80" height="210.2" fill="var(--warning)"/> <!-- 87.6% -->
    </g>
    <!-- value labels (optional) -->
    <g font-size="10" fill="var(--text-on-accent)" text-anchor="middle">
      <text x="120" y="140">79.5%</text>
      <text x="240" y="180">87.6%</text>
    </g>
    <!-- category labels -->
    <g font-size="11" fill="var(--text-secondary)" text-anchor="middle">
      <text x="120" y="296">A</text>
      <text x="240" y="296">B</text>
    </g>
  </svg>
  <div class="legend" style="margin-top:8px">
    <span><span class="swatch" style="background:var(--info)"></span>GEMM</span>
    <span><span class="swatch" style="background:var(--warning)"></span>NCCL</span>
  </div>
</div>
```

### BarChart — horizontal grouped

For `<BarChart horizontal …>` with multiple series, swap the axes: each
category is a horizontal row of two stacked-or-grouped `<rect>`s.

### Series-tone color mapping

| `tone=` on series | CSS var |
|---|---|
| (none) | `var(--chart-1)` then `--chart-2`, `--chart-3` … in order |
| `success` | `var(--success)` |
| `warning` | `var(--warning)` |
| `danger`  | `var(--danger)`  |
| `info`    | `var(--info)`    |
| `neutral` | `var(--text-tertiary)` |

### LineChart

Render a polyline per series. Pre-compute SVG x/y for each point given
the chart `width`, `height`, padding, and the series min/max.

```html
<svg viewBox="0 0 400 200" width="100%" height="200">
  <polyline fill="none" stroke="var(--chart-1)" stroke-width="1.5"
            points="40,160 130,100 220,130 310,40 400,80"/>
  <g fill="var(--chart-1)">
    <circle cx="40"  cy="160" r="3"/>
    <circle cx="130" cy="100" r="3"/>
    <circle cx="220" cy="130" r="3"/>
    <circle cx="310" cy="40"  r="3"/>
    <circle cx="400" cy="80"  r="3"/>
  </g>
</svg>
```

For `fill`, add a matching `<polygon>` underneath with reduced opacity:
`fill="var(--chart-1)" opacity="0.15"`.

### PieChart

For each slice, pre-compute the SVG arc path. For a small number of slices
this is faster than asking the user to stare at recompute math:

```html
<svg viewBox="0 0 200 200" width="200" height="200">
  <!-- slice 1: 70%, starts at top, sweep clockwise -->
  <path d="M100,100 L100,10 A90,90 0 1,1 22.27,154.27 Z"
        fill="var(--success)"/>
  <!-- slice 2: 30% -->
  <path d="M100,100 L22.27,154.27 A90,90 0 0,1 100,10 Z"
        fill="var(--danger)"/>
</svg>
```

Donut: add a centered `<circle cx="100" cy="100" r="50" fill="var(--bg)"/>`
on top.

If the source has many slices or non-trivial values, use this Python
snippet locally to compute paths, then paste the literal `d="…"` strings:

```python
import math
def slice_path(cx, cy, r, start_pct, sweep_pct):
    a0 = math.radians(start_pct * 3.6 - 90)
    a1 = math.radians((start_pct + sweep_pct) * 3.6 - 90)
    x0, y0 = cx + r*math.cos(a0), cy + r*math.sin(a0)
    x1, y1 = cx + r*math.cos(a1), cy + r*math.sin(a1)
    large = 1 if sweep_pct > 50 else 0
    return f"M{cx},{cy} L{x0:.2f},{y0:.2f} A{r},{r} 0 {large},1 {x1:.2f},{y1:.2f} Z"
```

## What NOT to convert

The following `cursor/canvas` exports are interactive or
Cursor-runtime-only. If you encounter them in the source, replace with a
**static substitute** and note it in the file as an HTML comment:

| Source component | Static substitute |
|---|---|
| `Button` (with `onClick`) | Drop entirely, or render as `<span class="pill">` (no action) |
| `Card collapsible` | Render expanded; drop the toggle |
| `DiffView` | Render as a `<pre>` with `+ ` / `- ` line prefixes, color via inline style |
| `DAGLayout` | Render the resolved layout as inline SVG (one-shot, not interactive) |
| `TodoList` | Render as a static `<ul>` with checkbox glyphs (`☐` / `☑`) |
| Any `useHostTheme()` outside an SVG | Use `var(--…)` in inline style |

If an interactive component is removed, add a one-line HTML comment at the
removal site explaining what was dropped, e.g.:

```html
<!-- Original canvas had a collapsible Card; rendered expanded for static export. -->
```
