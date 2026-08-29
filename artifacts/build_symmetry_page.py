"""Build artifacts/treecode_symmetry.html from the measured results.

Charts are emitted as inline SVG so the page is self-contained (the Artifact CSP blocks
every external host). Series colors are the dataviz reference palette slots 1-2, used
verbatim against its documented chart surfaces so its validation carries over.
"""

import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "treecode_symmetry.html")

# ---------------------------------------------------------------- measured data
THETA_N800 = [  # theta, rel_asym, trunc_err   (N=800, far-field block in isolation)
    (0.150, 2.417e-5, 1.709e-5),
    (0.175, 2.519e-4, 1.810e-4),
    (0.200, 9.766e-4, 7.064e-4),
    (0.250, 4.174e-3, 3.087e-3),
    (0.300, 9.033e-3, 6.755e-3),
    (0.400, 2.120e-2, 1.591e-2),
    (0.500, 3.700e-2, 2.785e-2),
    (0.600, 5.544e-2, 4.197e-2),
    (0.700, 7.381e-2, 5.538e-2),
]
ZERO_UNTIL = 0.125  # asymmetry identically zero at and below this theta

PARETO = [  # theta, rel_asym, far_ms, step_ms   (N=100k)
    (0.05, 4.663e-5, 236.27, 309.21),
    (0.10, 1.110e-3, 203.56, 274.71),
    (0.20, 4.100e-3, 49.75, 120.88),
    (0.30, 1.385e-2, 18.71, 88.76),
    (0.40, 2.656e-2, 8.93, 79.24),
    (0.50, 4.491e-2, 5.20, 75.29),
]

SCALE = [  # N, rel_asym, stderr   (theta=0.3)
    (150, 1.1922e-3, 0.0),
    (10_000, 1.0997e-2, 1.7038e-3),
    (100_000, 1.2544e-2, 1.7610e-3),
    (1_000_000, 1.5271e-2, 2.4767e-3),
]

BLOCKS = [  # label, frobenius norm, kept?
    ("TT  translation ← force", 1.4149, True),
    ("RT  translation ← torque", 0.10018, False),
    ("TR  rotation ← force", 0.10018, False),
    ("RR  rotation ← torque", 0.011534, False),
]

# ---------------------------------------------------------------- scale helpers
def linscale(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def logscale(v, lo, hi, a, b):
    return a + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (b - a)


def sci(v, digits=2):
    """1.23 x 10^-4 as SVG-safe markup-free text."""
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    m = v / 10 ** e
    return f"{m:.{digits}f}×10{sup(e)}"


SUPS = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
        "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸",
        "9": "⁹"}


def sup(n):
    return "".join(SUPS[c] for c in str(n))


def decade_ticks(lo, hi):
    out = []
    e = math.ceil(math.log10(lo))
    while 10 ** e <= hi * 1.0001:
        out.append(10 ** e)
        e += 1
    return out


# ---------------------------------------------------------------- chart 1
def chart_theta():
    W, H = 880, 400
    L, R, T, B = 66, 128, 26, 52
    x0, x1 = 0.0, 0.74
    y0, y1 = 1e-5, 1.2e-1

    X = lambda v: linscale(v, x0, x1, L, W - R)
    Y = lambda v: linscale(math.log10(v), math.log10(y0), math.log10(y1), H - B, T)

    p = []
    p.append(f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Relative asymmetry and truncation error versus opening angle theta">')

    # zero region
    p.append(f'<rect x="{L}" y="{T}" width="{X(ZERO_UNTIL)-L:.1f}" height="{H-B-T}" '
             f'class="band"/>')
    p.append(f'<text x="{(L+X(ZERO_UNTIL))/2:.1f}" y="{T+16}" class="band-lab" '
             f'text-anchor="middle">identically</text>')
    p.append(f'<text x="{(L+X(ZERO_UNTIL))/2:.1f}" y="{T+30}" class="band-lab" '
             f'text-anchor="middle">zero</text>')

    # grid + y ticks
    for t in decade_ticks(y0, y1):
        p.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{W-R}" y2="{Y(t):.1f}" class="grid"/>')
        p.append(f'<text x="{L-10}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">'
                 f'10{sup(int(round(math.log10(t))))}</text>')
    # x ticks
    for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        p.append(f'<line x1="{X(t):.1f}" y1="{H-B}" x2="{X(t):.1f}" y2="{H-B+5}" class="axis"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{t:g}</text>')
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" class="axis"/>')

    # the two series converge to ~10 px apart at the right edge, so the direct labels
    # are pushed apart vertically rather than sitting at their endpoints
    for key, cls, name, ldy in ((1, "s1", "asymmetry", -10), (2, "s2", "truncation error", 18)):
        pts = [(X(r[0]), Y(r[key])) for r in THETA_N800]
        d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
        p.append(f'<path d="{d}" class="line {cls}"/>')
        for (a, b), r in zip(pts, THETA_N800):
            p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4.5" class="dot {cls}">'
                     f'<title>theta {r[0]:g} — {name} {sci(r[key])}</title></circle>')
        lx, ly = pts[-1]
        p.append(f'<line x1="{lx+5:.1f}" y1="{ly:.1f}" x2="{lx+10:.1f}" y2="{ly+ldy-4:.1f}" '
                 f'class="leader {cls}"/>')
        p.append(f'<text x="{lx+13:.1f}" y="{ly+ldy:.1f}" class="dlab {cls}">{name}</text>')

    # production marker
    p.append(f'<line x1="{X(0.3):.1f}" y1="{T}" x2="{X(0.3):.1f}" y2="{H-B}" class="marker"/>')
    p.append(f'<text x="{X(0.3):.1f}" y="{T-8}" class="marker-lab" text-anchor="middle">'
             f'production θ = 0.3</text>')

    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-8}" class="axlab" text-anchor="middle">'
             f'opening angle θ</text>')
    p.append(f'<text transform="translate(16,{(T+H-B)/2:.1f}) rotate(-90)" class="axlab" '
             f'text-anchor="middle">relative Frobenius norm</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- chart 2
def chart_pareto():
    W, H = 880, 400
    L, R, T, B = 66, 60, 34, 52
    x0, x1 = 68.0, 360.0
    y0, y1 = 3e-5, 8e-2

    X = lambda v: linscale(math.log10(v), math.log10(x0), math.log10(x1), L, W - R)
    Y = lambda v: linscale(math.log10(v), math.log10(y0), math.log10(y1), H - B, T)

    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Relative asymmetry versus full step time at N equals 100,000">']

    for t in decade_ticks(y0, y1):
        p.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{W-R}" y2="{Y(t):.1f}" class="grid"/>')
        p.append(f'<text x="{L-10}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">'
                 f'10{sup(int(round(math.log10(t))))}</text>')
    for t in (75, 100, 150, 200, 300):
        p.append(f'<line x1="{X(t):.1f}" y1="{H-B}" x2="{X(t):.1f}" y2="{H-B+5}" class="axis"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{t}</text>')
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" class="axis"/>')

    pts = [(X(r[3]), Y(r[1])) for r in PARETO]
    d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
    p.append(f'<path d="{d}" class="line s1"/>')

    for (a, b), r in zip(pts, PARETO):
        th, asym, far, step = r
        hi = th in (0.20, 0.30)
        cls = "s2" if th == 0.30 else ("s1 rec" if th == 0.20 else "s1")
        p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{7 if hi else 5}" class="dot {cls}">'
                 f'<title>theta {th:g} — asymmetry {sci(asym)}, far field {far:.1f} ms, '
                 f'full step {step:.1f} ms</title></circle>')
        p.append(f'<text x="{a:.1f}" y="{b-16:.1f}" class="ptlab{" em" if hi else ""}" '
                 f'text-anchor="middle">θ {th:g}</text>')

    # annotations go below their point so they never collide with the theta labels above
    ax, ay = X(88.76), Y(1.385e-2)
    p.append(f'<text x="{ax:.1f}" y="{ay+24:.1f}" class="note s2" '
             f'text-anchor="middle">current</text>')
    bx, by = X(120.88), Y(4.100e-3)
    p.append(f'<text x="{bx:.1f}" y="{by+24:.1f}" class="note rec" '
             f'text-anchor="middle">recommended</text>')

    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-8}" class="axlab" text-anchor="middle">'
             f'full mobility step (ms, N = 100,000)</text>')
    p.append(f'<text transform="translate(16,{(T+H-B)/2:.1f}) rotate(-90)" class="axlab" '
             f'text-anchor="middle">relative asymmetry</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- chart 3
def chart_scale():
    W, H = 880, 340
    L, R, T, B = 66, 60, 26, 52
    x0, x1 = 80.0, 3e6
    y0, y1 = 8e-4, 3e-2

    X = lambda v: linscale(math.log10(v), math.log10(x0), math.log10(x1), L, W - R)
    Y = lambda v: linscale(math.log10(v), math.log10(y0), math.log10(y1), H - B, T)

    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Relative asymmetry versus particle count at theta 0.3">']
    for t in decade_ticks(y0, y1):
        p.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{W-R}" y2="{Y(t):.1f}" class="grid"/>')
        p.append(f'<text x="{L-10}" y="{Y(t)+4:.1f}" class="tick" text-anchor="end">'
                 f'10{sup(int(round(math.log10(t))))}</text>')
    for t, lab in ((100, "10²"), (1e3, "10³"), (1e4, "10⁴"),
                   (1e5, "10⁵"), (1e6, "10⁶")):
        p.append(f'<line x1="{X(t):.1f}" y1="{H-B}" x2="{X(t):.1f}" y2="{H-B+5}" class="axis"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-B+20}" class="tick" text-anchor="middle">{lab}</text>')
    p.append(f'<line x1="{L}" y1="{H-B}" x2="{W-R}" y2="{H-B}" class="axis"/>')

    pts = [(X(n), Y(v)) for n, v, _ in SCALE]
    d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
    p.append(f'<path d="{d}" class="line s1"/>')
    for (a, b), (n, v, se) in zip(pts, SCALE):
        if se > 0:
            p.append(f'<line x1="{a:.1f}" y1="{Y(max(v-se,y0*1.02)):.1f}" x2="{a:.1f}" '
                     f'y2="{Y(v+se):.1f}" class="err"/>')
        p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" class="dot s1">'
                 f'<title>N {n:,} — asymmetry {sci(v)}</title></circle>')
        p.append(f'<text x="{a:.1f}" y="{b-16:.1f}" class="ptlab" text-anchor="middle">'
                 f'{v*100:.2f}%</text>')
    p.append(f'<text x="{X(150):.1f}" y="{Y(1.1922e-3)+18:.1f}" class="note" '
             f'text-anchor="middle">dense assembly</text>')

    p.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-8}" class="axlab" text-anchor="middle">'
             f'particles N</text>')
    p.append(f'<text transform="translate(16,{(T+H-B)/2:.1f}) rotate(-90)" class="axlab" '
             f'text-anchor="middle">relative asymmetry</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- chart 4
def chart_blocks():
    W = 880
    rowh, gap = 42, 12
    L, R, T = 190, 200, 26   # R leaves room for the value + "% of TT" label past the bar
    H = T + len(BLOCKS) * (rowh + gap) + 34
    ref = BLOCKS[0][1]
    xmax = W - R

    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Far-field operator block norms, kept versus discarded">']
    for i, (lab, val, kept) in enumerate(BLOCKS):
        y = T + i * (rowh + gap)
        w = (val / ref) * (xmax - L)
        cls = "s1" if kept else "s2"
        p.append(f'<text x="{L-14}" y="{y+rowh/2+5:.0f}" class="rowlab" text-anchor="end">'
                 f'{lab}</text>')
        p.append(f'<rect x="{L}" y="{y}" width="{max(w,2.5):.1f}" height="{rowh}" rx="4" '
                 f'class="bar {cls}"><title>{lab} — Frobenius norm {val:.4f}, '
                 f'{"kept" if kept else "discarded"}</title></rect>')
        p.append(f'<text x="{L+max(w,2.5)+12:.1f}" y="{y+rowh/2+5:.0f}" class="barval">'
                 f'{val:.4f}<tspan class="barpct">  ({val/ref*100:.1f}% of TT)</tspan></text>')
    p.append(f'<text x="{L}" y="{H-8}" class="note">'
             f'blue = evaluated by the treecode · orange = set to zero beyond r = 6</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------- table views
def table(headers, rows, caption):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="tableview"><summary>{caption}</summary>'
            f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody>'
            f"</table></div></details>")


CSS = """
:root {
  color-scheme: light;
  --plane:#f2f3f6; --card:#ffffff; --chart-surface:#fcfcfb;
  --ink:#14161c; --ink-2:#4b505c; --muted:#7f848f;
  --rule:#dfe2e9; --rule-soft:#eaecf1;
  --accent:#27478d; --accent-soft:#e8ecf7;
  --s1:#2a78d6; --s2:#eb6834;
  --grid:#e6e8ee; --axis:#c3c7d0;
  --crit:#d03b3b; --good:#0ca30c;
  --band:#f0f2f7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane:#0e0f13; --card:#171920; --chart-surface:#1a1a19;
    --ink:#f0f2f6; --ink-2:#b3b9c6; --muted:#868c99;
    --rule:#2b2e37; --rule-soft:#22242b;
    --accent:#87a8ee; --accent-soft:#1b2333;
    --s1:#3987e5; --s2:#d95926;
    --grid:#2a2c31; --axis:#3a3d46;
    --crit:#e36a6a; --good:#38b93a;
    --band:#20232b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane:#0e0f13; --card:#171920; --chart-surface:#1a1a19;
  --ink:#f0f2f6; --ink-2:#b3b9c6; --muted:#868c99;
  --rule:#2b2e37; --rule-soft:#22242b;
  --accent:#87a8ee; --accent-soft:#1b2333;
  --s1:#3987e5; --s2:#d95926;
  --grid:#2a2c31; --axis:#3a3d46;
  --crit:#e36a6a; --good:#38b93a;
  --band:#20232b;
}

* { box-sizing: border-box; }
body {
  margin:0; background:var(--plane); color:var(--ink);
  font-family: ui-serif, Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
  font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap { max-width:960px; margin:0 auto; padding:56px 24px 96px; }
.prose { max-width:68ch; }

h1,h2,h3,.eyebrow,.tick,.axlab,.dlab,.ptlab,.rowlab,.note,.band-lab,.marker-lab,
.stat-lab,.pill,summary,th,.kv {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.eyebrow {
  font-size:12px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); font-weight:600; margin:0 0 14px;
}
h1 { font-size:clamp(30px,4.4vw,44px); line-height:1.12; letter-spacing:-.022em;
     font-weight:640; margin:0 0 18px; text-wrap:balance; }
h2 { font-size:22px; letter-spacing:-.01em; font-weight:640; margin:0 0 4px; }
h3 { font-size:16px; font-weight:640; margin:28px 0 6px; letter-spacing:-.005em; }
.lede { font-size:19.5px; color:var(--ink-2); margin:0 0 8px; }
p { margin:0 0 18px; }
a { color:var(--accent); text-underline-offset:3px; }

.meta {
  display:flex; flex-wrap:wrap; gap:8px 22px; margin:26px 0 0;
  padding-top:18px; border-top:1px solid var(--rule);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12.5px; color:var(--muted);
}

/* verdict */
.verdict { margin:40px 0 8px; display:grid; gap:14px;
           grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); }
.stat { background:var(--card); border:1px solid var(--rule); border-radius:10px;
        padding:20px 20px 18px; }
.stat .v { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
           font-size:34px; font-weight:600; letter-spacing:-.02em; line-height:1.05;
           font-variant-numeric: tabular-nums; }
.stat .v.bad { color:var(--crit); }
.stat .v.ok  { color:var(--good); }
.stat-lab { font-size:13px; color:var(--ink-2); margin-top:8px; line-height:1.42; }

.callout {
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:16px 20px; border-radius:0 8px 8px 0; margin:26px 0;
}
.callout p:last-child { margin-bottom:0; }

section { margin-top:52px; }
.sechead { display:flex; align-items:baseline; gap:12px; margin-bottom:16px;
           padding-bottom:10px; border-bottom:1px solid var(--rule); }
.snum { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size:13px; color:var(--muted); font-weight:600; }

/* figures */
figure { margin:26px 0 8px; background:var(--card); border:1px solid var(--rule);
         border-radius:10px; padding:18px 18px 12px; }
figcaption { font-size:13.5px; color:var(--ink-2); margin-top:6px;
             padding-top:12px; border-top:1px solid var(--rule-soft); }
.chart { display:block; width:100%; height:auto; background:var(--chart-surface);
         border-radius:6px; }
.grid { stroke:var(--grid); stroke-width:1; }
.axis { stroke:var(--axis); stroke-width:1; }
.band { fill:var(--band); }
.band-lab { fill:var(--muted); font-size:11px; }
.tick { fill:var(--muted); font-size:11.5px; font-variant-numeric:tabular-nums; }
.axlab { fill:var(--ink-2); font-size:12.5px; }
.line { fill:none; stroke-width:2; }
.line.s1 { stroke:var(--s1); } .line.s2 { stroke:var(--s2); }
.dot { stroke:var(--chart-surface); stroke-width:2; }
.dot.s1 { fill:var(--s1); } .dot.s2 { fill:var(--s2); }
.dot.rec { fill:var(--s1); stroke:var(--s1); stroke-width:3; }
.err { stroke:var(--s1); stroke-width:1.5; opacity:.55; }
.dlab { font-size:12.5px; font-weight:600; }
.dlab.s1 { fill:var(--s1); } .dlab.s2 { fill:var(--s2); }
.leader { stroke-width:1; }
.leader.s1 { stroke:var(--s1); } .leader.s2 { stroke:var(--s2); }
.ptlab { fill:var(--ink-2); font-size:11.5px; font-variant-numeric:tabular-nums; }
.ptlab.em { fill:var(--ink); font-weight:600; }
.note { fill:var(--muted); font-size:11.5px; }
.note.s2 { fill:var(--s2); font-weight:600; }
.note.rec { fill:var(--s1); font-weight:600; }
.marker { stroke:var(--axis); stroke-width:1; stroke-dasharray:3 4; }
.marker-lab { fill:var(--muted); font-size:11px; }
.rowlab { fill:var(--ink-2); font-size:13px; }
.bar.s1 { fill:var(--s1); } .bar.s2 { fill:var(--s2); }
.barval { fill:var(--ink); font-size:13px;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-variant-numeric:tabular-nums; }
.barpct { fill:var(--muted); }
.chart circle:hover { stroke-width:3; }
.chart rect.bar:hover { opacity:.85; }

/* tables */
.tableview { margin-top:10px; }
summary { cursor:pointer; font-size:12.5px; color:var(--muted); padding:6px 0; }
summary:hover { color:var(--ink-2); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-variant-numeric:tabular-nums; }
th { text-align:left; font-size:12px; letter-spacing:.04em; text-transform:uppercase;
     color:var(--muted); font-weight:600; padding:8px 14px 8px 0;
     border-bottom:1px solid var(--rule); white-space:nowrap; }
td { padding:7px 14px 7px 0; border-bottom:1px solid var(--rule-soft);
     color:var(--ink-2); white-space:nowrap; }
td:first-child { color:var(--ink); }
tr.em td { color:var(--ink); font-weight:600; }

code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size:.88em; }
code { background:var(--rule-soft); padding:1.5px 5px; border-radius:4px; }
pre { background:var(--card); border:1px solid var(--rule); border-radius:8px;
      padding:16px 18px; overflow-x:auto; font-size:13px; line-height:1.6; }
pre code { background:none; padding:0; }

ol.fixes { list-style:none; counter-reset:f; padding:0; margin:22px 0 0; }
ol.fixes > li { counter-increment:f; position:relative; padding:18px 0 18px 46px;
                border-top:1px solid var(--rule); }
ol.fixes > li::before {
  content:counter(f); position:absolute; left:0; top:20px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;
  font-weight:600; color:var(--accent); background:var(--accent-soft);
  width:26px; height:26px; border-radius:50%; display:grid; place-items:center;
}
ol.fixes h3 { margin:0 0 6px; }
ol.fixes p { margin:0 0 8px; font-size:16px; }
ol.fixes p:last-child { margin-bottom:0; }
.pill { display:inline-block; font-size:11px; font-weight:600; letter-spacing:.05em;
        text-transform:uppercase; padding:2.5px 8px; border-radius:999px;
        border:1px solid var(--rule); color:var(--muted); margin-left:8px;
        vertical-align:2px; }
.pill.rec { color:var(--good); border-color:var(--good); }
ul.plain { padding-left:20px; }
ul.plain li { margin-bottom:12px; }
"""


def build():
    rows_theta = [(f"{t:g}", sci(a), sci(u), f"{a/u:.3f}") for t, a, u in THETA_N800]
    rows_theta = ([("0.000 – 0.125", "0 (exact)", "8.16×10⁻⁸", "—")]
                  + rows_theta)
    t1 = table(["θ", "rel. asymmetry", "truncation error", "ratio"], rows_theta,
               "Table view — asymmetry and truncation error vs θ (N = 800)")
    t2 = table(["θ", "rel. asymmetry", "far field", "full step"],
               [(f"{t:g}", sci(a), f"{f:.1f} ms", f"{s:.1f} ms") for t, a, f, s in PARETO],
               "Table view — cost and accuracy vs θ (N = 100,000)")
    t3 = table(["N", "rel. asymmetry", "std. error"],
               [(f"{n:,}", sci(v), sci(se) if se else "dense") for n, v, se in SCALE],
               "Table view — asymmetry vs system size (θ = 0.3)")
    t4 = table(["block", "‖·‖_F", "% of TT", "treecode"],
               [(l.split("  ")[0] + "  " + l.split("  ")[1], f"{v:.4f}",
                 f"{v/BLOCKS[0][1]*100:.2f}%", "kept" if k else "discarded")
                for l, v, k in BLOCKS],
               "Table view — far-field operator block norms (N = 150)")

    html = f"""<title>Is the NeMO grand mobility symmetric with the treecode?</title>
<style>{CSS}</style>
<div class="wrap">

<p class="eyebrow">NeMO mobility operator &middot; numerical verification</p>
<h1>Is the grand mobility still symmetric once the treecode handles the far field?</h1>
<p class="lede prose">No. But the near field is, the cause is provable, and the asymmetry
turns out to be the treecode's own truncation error rather than a defect on top of it.
A larger problem showed up alongside it.</p>

<div class="meta">
  <span>benchmarks/symmetry_treecode.py</span>
  <span>NVIDIA H200</span>
  <span>TORCH_COMPILE_DISABLE=1</span>
  <span>branch: performance</span>
</div>

<div class="verdict">
  <div class="stat">
    <div class="v bad">1.5%</div>
    <div class="stat-lab">relative asymmetry of the grand mobility at
      N&nbsp;=&nbsp;1&nbsp;M, production &#952;&nbsp;=&nbsp;0.3</div>
  </div>
  <div class="stat">
    <div class="v ok">4&#215;10&#8315;&#8313;</div>
    <div class="stat-lab">the same operator with the treecode off &mdash; so the tree is
      the entire cause, not the neural near field</div>
  </div>
  <div class="stat">
    <div class="v bad">10.0%</div>
    <div class="stat-lab">of the far-field operator's Frobenius norm is discarded: it
      carries no rotation&ndash;translation coupling at all</div>
  </div>
</div>

<div class="callout prose">
  <p><strong>What to do.</strong> The symmetry defect is real but second-order. The
  translation-only far field is <strong>6&#215; larger, systematic, and makes particle
  rotation rates untrustworthy</strong> in any treecode run &mdash; fix that first. For
  symmetry itself, dropping &#952; from 0.3 to 0.2 costs 36&nbsp;% step time and buys
  3.4&#215; on both asymmetry and far-field accuracy; a symmetrized apply is exact and
  costs about the same.</p>
</div>

<section>
  <div class="sechead"><span class="snum">01</span><h2>The near field is symmetric</h2></div>
  <div class="prose">
  <p><code>Mob_Nbody_Torch</code> with <code>far_field_2b=None</code> &mdash; self plus
  two-body NN plus n-body NN, nothing else &mdash; assembled column by column into a
  900&#215;900 matrix gives
  <span class="mono">&#8214;M&#8722;M&#7488;&#8214;/&#8214;M&#8214; = 4.85&#215;10&#8315;&#8313;</span>
  and <span class="mono">max|M&#8722;M&#7488;| = 1.40&#215;10&#8315;&#8313;</span>. Symmetric
  to float32 precision.</p>
  <p>Worth recording because <code>src/model_archs.py</code> warns that the two-body kernel
  <span class="mono">M_t</span> is deliberately <em>not</em> symmetric. That turns out to be
  exactly the condition that makes the assembled matrix symmetric &mdash; with
  <span class="mono">RT = c(r)&middot;L3(d&#770;)</span> and
  <span class="mono">L3(&#8722;d&#770;) = &#8722;L3(d&#770;) = L3(d&#770;)&#7488;</span>, the
  directed-edge scatter in <code>PairVelKernel</code> lands the transpose in the right
  place on its own.</p>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">02</span><h2>The far field is not &mdash; and the
    multipole acceptance is provably the cause</h2></div>
  <figure>
    {chart_theta()}
    <figcaption>Far-field block assembled in isolation (N&nbsp;=&nbsp;800, 2400&#215;2400)
      against a dense float64 RPY reference over all pairs with r&nbsp;&#8805;&nbsp;6. Below
      &#952;&nbsp;&#8776;&nbsp;0.15 the asymmetry is identically zero &mdash; not small,
      zero.</figcaption>
    {t1}
  </figure>
  <div class="prose">
  <p><strong>The tree is the cause.</strong> Below &#952;&nbsp;&#8776;&nbsp;0.15 no node
  passes the acceptance test, traversal degenerates to all-direct pairs, and the result is
  bitwise symmetric: <code>rpy_far_velocity_pair3x3</code> builds a tensor that is symmetric
  and even in <strong>r</strong>, and with a unit force on a single particle there is no
  summation to reorder. Asymmetry switches on exactly when the tree starts lumping
  sources.</p>
  <p><strong>The near/far partition is clean.</strong> At &#952;&nbsp;=&nbsp;0 the assembled
  block matches the dense reference to 8&#215;10&#8315;&#8312;. The
  <code>closest_dist_sq &lt; 36.0</code> guard in <code>warp/native/bvh.h:640</code> and the
  <code>dot(rvec,rvec) &lt; near_cutoff2</code> skip in <code>treecode.py:178</code> really do
  produce the exact r&nbsp;&#8805;&nbsp;6 partition &mdash; no double counting, no gaps.</p>
  <p><strong>The asymmetry <em>is</em> the truncation error.</strong> The ratio between them
  sits at &#8730;2&nbsp;=&nbsp;1.414 at onset and drifts only to about 1.33. If the treecode
  error <span class="mono">E = M &#8722; M_exact</span> were uncorrelated with its own
  transpose you would get exactly &#8730;2; landing at or just under it means essentially
  none of <span class="mono">E</span> is a symmetric common mode. There is no separate
  asymmetry problem to solve &mdash; shrink the truncation error and the asymmetry shrinks
  one-for-one.</p>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">03</span><h2>Treecode on versus off</h2></div>
  <div class="prose">
  <p>Full 900&#215;900 grand mobility at N&nbsp;=&nbsp;150, &#952;&nbsp;=&nbsp;0.3, assembled
  both ways:</p>
  </div>
  <figure>
    <div class="scroll"><table>
      <thead><tr><th>far field</th><th>rel. asymmetry</th><th>max|M&#8722;M&#7488;|</th>
        <th>neg. eigenvalues</th><th>min eigenvalue</th></tr></thead>
      <tbody>
        <tr class="em"><td>treecode</td><td>1.19&#215;10&#8315;&#179;</td>
          <td>2.42&#215;10&#8315;&#8308;</td><td>33</td><td>&#8722;9.565&#215;10&#8315;&#179;</td></tr>
        <tr><td>dense analytic RPY</td><td>3.77&#215;10&#8315;&#8313;</td>
          <td>1.16&#215;10&#8315;&#8313;</td><td>33</td><td>&#8722;9.428&#215;10&#8315;&#179;</td></tr>
      </tbody>
    </table></div>
    <figcaption>Five orders of magnitude separate the two rows on symmetry &mdash; and the
      negative-eigenvalue count is identical.</figcaption>
  </figure>
  <div class="prose">
  <p>Two conclusions. The treecode is the <strong>sole source of asymmetry</strong>. And the
  treecode is <strong>not</strong> the source of the negative eigenvalues: both operators
  have exactly 33, with near-identical minima. The grand mobility is already non-SPD without
  any tree &mdash; that comes from the neural near field and is pre-existing.</p>
  <p>(The far-field-free operator in &#167;01 shows 110 negative eigenvalues, but that is an
  artifact of hard-truncating the far field to zero, which cannot preserve
  positive-definiteness. It is not an operator anyone runs.)</p>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">04</span><h2>At production scale</h2></div>
  <figure>
    {chart_scale()}
    <figcaption>Hutchinson probes &mdash;
      <span class="mono">E[(u&#7488;Mv &#8722; v&#7488;Mu)&#178;] = &#8214;M&#8722;M&#7488;&#8214;&#178;</span>
      for Rademacher u,&nbsp;v &mdash; so no assembly is needed at 10&#8309;&#8211;10&#8310;
      particles. Validated first against the dense N&nbsp;=&nbsp;150 result (probe
      1.35&#215;10&#8315;&#179;&nbsp;&#177;&nbsp;2.1&#215;10&#8315;&#8308; vs. dense
      1.19&#215;10&#8315;&#179;, inside one standard error). Bars are &#177;1 s.e. over
      K&nbsp;=&nbsp;24 probes.</figcaption>
    {t3}
  </figure>
  <div class="prose">
  <p>Asymmetry grows with N and then plateaus near 1.5&nbsp;%: deeper trees serve more of the
  far field from lumped nodes, but that fraction saturates.</p>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">05</span><h2>What it costs to buy symmetry back</h2></div>
  <figure>
    {chart_pareto()}
    <figcaption>N&nbsp;=&nbsp;100,000 on the H200. The near field costs roughly 70&nbsp;ms of
      every step regardless of &#952;, which is what makes the far field cheap to buy
      down.</figcaption>
    {t2}
  </figure>
  <div class="prose">
  <p><strong>&#952;&nbsp;0.3&nbsp;&rarr;&nbsp;0.2 cuts asymmetry 3.4&#215; for 1.36&#215; the
  step time</strong> &mdash; and cuts the far-field truncation error by the same 3.4&#215;,
  since &#167;02 showed the two move together. Going further to &#952;&nbsp;=&nbsp;0.1 costs
  3.1&#215; the step time for 12&#215;, which is a much worse trade.</p>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">06</span><h2>The bigger finding: the far field
    carries no rotation</h2></div>
  <div class="prose">
  <p><code>WarpFMM.apply</code> slices <code>forces[:, :3]</code> and writes back into
  <code>total_vel[:, :3]</code>. Torques never enter the far field, and the RT, TR and RR
  blocks are exactly zero beyond r&nbsp;=&nbsp;6. That is symmetric &mdash; a zero block is
  its own transpose &mdash; but it is a real physics truncation, and it is larger than the
  symmetry defect.</p>
  </div>
  <figure>
    {chart_blocks()}
    <figcaption>Dense full-6&#215;6 far-field operator, N&nbsp;=&nbsp;150, validated against
      <code>benchmarks/bench_rpy.py:two_body_rpy_batch</code> to 5&#215;10&#8315;&#8312;.
      Discarding RT, TR and RR removes <strong>10.0&nbsp;% of
      &#8214;M_far&#8214;_F</strong> &mdash; which accounts almost exactly for the 5.8&nbsp;%
      gap between the treecode and dense-RPY grand matrices in &#167;03.</figcaption>
    {t4}
  </figure>
  <div class="prose">
  <p>The damage concentrates in angular velocity, and specifically in the <strong>TR block
  &mdash; rotation driven by neighbours' forces</strong>, not by their torques. Under a
  sedimentation-style loading (uniform gravity plus random torques) the discarded far-field
  angular velocity is <em>larger than the entire near-field angular velocity</em>:</p>
  </div>
  <figure>
    <div class="scroll"><table>
      <thead><tr><th>config</th><th>&#8214;&#937;_far&#8214; / &#8214;&#937;_near&#8214;</th>
        <th>&#937;_far from forces</th><th>&#937;_far from torques</th>
        <th>U error</th></tr></thead>
      <tbody>
        <tr><td>N = 800, &#966; &#8776; 0.1</td><td>1.17</td><td>2.012</td><td>0.029</td>
          <td>3.25&#215;10&#8315;&#179;</td></tr>
        <tr><td>N = 10,000, &#966; &#8776; 0.1</td><td>1.74</td><td>11.33</td><td>0.099</td>
          <td>8.01&#215;10&#8315;&#8308;</td></tr>
      </tbody>
    </table></div>
    <figcaption>99&nbsp;% of the discarded rotation comes from the force&rarr;rotation
      coupling. Translational velocity is barely affected.</figcaption>
  </figure>
  <div class="prose">
  <p>So: linear dynamics are fine. <strong>Particle rotation rates in a treecode run are
  not.</strong></p>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">07</span><h2>Options</h2></div>
  <ol class="fixes">
    <li>
      <h3>Lower theta<span class="pill rec">free today</span></h3>
      <p>&#952;&nbsp;=&nbsp;0.2 buys 3.4&#215; on both asymmetry and far-field accuracy for
      36&nbsp;% more wall time. Nothing to implement &mdash; it is a constructor
      argument.</p>
    </li>
    <li>
      <h3>Symmetrized apply, u = &#189;(Mf + M&#7488;f)</h3>
      <p><span class="mono">M&#7488;f</span> needs no assembly: re-run the <em>same</em>
      traversal and scatter with atomics &mdash; for each source j inside an accepted node of
      target i, add <span class="mono">M_eff[i,j]&#7488; f_i</span> into
      <span class="mono">v_j</span>. Identical traversal means an identical pair set, so the
      result is symmetric by construction, exactly, at any &#952;.</p>
      <p>Cost is about 2&#215; the far field plus atomic contention &mdash; at
      &#952;&nbsp;=&nbsp;0.3 that is roughly +19&nbsp;ms on an 89&nbsp;ms step, cheaper than
      dropping to &#952;&nbsp;=&nbsp;0.2. It is the only option giving <em>exact</em> symmetry
      without touching the vendored Warp BVH. It does not improve accuracy &mdash; the
      truncation error stays.</p>
    </li>
    <li>
      <h3>Dual-tree traversal with a symmetric node&ndash;node criterion and M2L</h3>
      <p>The principled fix: a reciprocal partition and, with matched truncation order on
      both sides, close to self-adjoint. A large rewrite of
      <code>warp/native/bvh.{{h,cu}}</code>, and worth noting that even textbook FMM with M2L
      is not <em>exactly</em> symmetric &mdash; so this buys accuracy more than it buys
      symmetry.</p>
    </li>
    <li>
      <h3>Accept it</h3>
      <p>Defensible for deterministic dynamics: 1.5&nbsp;% sits inside the treecode's own
      truncation budget and is dwarfed by &#167;06. <strong>Not</strong> defensible for
      Brownian or fluctuating hydrodynamics, where a symmetric PSD M is needed to form
      <span class="mono">M&#185;&#8260;&#178;</span> (Cholesky or Lanczos) and to satisfy
      fluctuation&ndash;dissipation. This operator is neither symmetric nor PSD.</p>
    </li>
  </ol>
</section>

<section>
  <div class="sechead"><span class="snum">08</span><h2>Found along the way</h2></div>
  <div class="prose">
  <ul class="plain">
    <li><strong><code>near_field_cutoff</code> is a live footgun.</strong> It is a
      <code>WarpFMM</code> constructor argument, but the matching gate is hardcoded as
      <code>36.0</code> in <code>warp/native/bvh.h:640</code>. Any value other than 6.0
      silently double-counts pairs (below 6) or drops them (above 6). An
      <code>assert near_field_cutoff == 6.0</code> would close it.</li>
    <li><strong>Viscosity is ignored inside <code>WarpFMM.apply</code>.</strong> The
      far-field kernels are called with <code>a=1.0, mu=1.0</code> hardcoded
      (<code>treecode.py:172,181</code>) and <code>_gpu_near_field_pass</code> is invoked with
      <code>viscosity=1.0</code> (<code>:426</code>), so the <code>vis_arr</code> argument
      does nothing. Silent wrong answer for &#956;&nbsp;&#8800;&nbsp;1.</li>
    <li><strong><code>src/grpy.py</code> has two bugs.</strong>
      <code>tr_block = -muRT_flat.T</code> (<code>:124</code>) makes the flat (6N,6N) output
      non-symmetric and disagrees with its own <code>blockmatrix=True</code> branch; and
      <code>_build_cross</code> (<code>:136</code>) has the opposite rotlet sign from the
      PyGRPY reference in <code>src/grpy_tensors.py:14</code>. No module imports it today,
      which is why it has gone unnoticed. The production paths
      (<code>benchmarks/bench_rpy.py</code>, <code>src/grpy_tensors.py</code>) are both
      correct and symmetric.</li>
  </ul>
  </div>
</section>

<section>
  <div class="sechead"><span class="snum">09</span><h2>Reproducing</h2></div>
  <pre><code>source ~/warp_env.sh
export TORCH_COMPILE_DISABLE=1

python benchmarks/symmetry_treecode.py near                 # 01
python benchmarks/symmetry_treecode.py theta --n-far 800    # 02
python benchmarks/symmetry_treecode.py grand --n 150        # 03
python benchmarks/symmetry_treecode.py probe --K 24         # 04
python benchmarks/symmetry_treecode.py thetascale --K 16    # 05
python benchmarks/symmetry_treecode.py ttonly               # 06
python benchmarks/symmetry_treecode.py all                  # everything -> artifacts/*.json</code></pre>
  <div class="prose">
  <p>Three self-checks run before any number is reported: the dense float64 reference must be
  exactly symmetric, the &#952;&nbsp;=&nbsp;0 far field must match it to float32, and the
  Hutchinson estimator must reproduce the dense answer at N&nbsp;=&nbsp;150. All three
  pass.</p>
  </div>
</section>

</div>
"""
    with open(OUT, "w") as fh:
        fh.write(html)
    print(f"wrote {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    build()
