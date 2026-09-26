"""Render docs/results/bench_envs.csv as an SVG, with no plotting dependency.

    python -m scripts.bench_chart

Writes docs/media/bench_envs.svg: a stacked bar per n_envs showing where a vec step goes, plus the throughput
curve. Hand-rolled SVG because the project has no plotting library and adding one for a single chart is not
worth a dependency.
"""
from __future__ import annotations

import csv
from pathlib import Path

CSV = Path("docs/results/bench_envs.csv")
OUT = Path("docs/media/bench_envs.svg")
W, H = 900, 380
PAD_L, PAD_R, PAD_T, PAD_B = 62, 18, 46, 52
PARTS = [("emulator_s", "emulator step", "#4C78A8"),
         ("obs_build_s", "observation build", "#54A24B"),
         ("policy_forward_s", "policy forward", "#E45756"),
         ("ipc_contention_s", "IPC + contention", "#F58518")]


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    n = [int(r["n_envs"]) for r in rows]
    stacks = [[max(0.0, float(r[k]) * 1e3) for k, _, _ in PARTS] for r in rows]
    sps = [float(r["agent_steps_per_s"]) for r in rows]

    plot_w = (W - PAD_L - PAD_R) / 2 - 24
    plot_h = H - PAD_T - PAD_B
    y_max = max(sum(s) for s in stacks) * 1.12
    s_max = max(sps) * 1.12
    bar_w = plot_w / (len(n) * 1.6)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="system-ui,sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{W/2}" y="20" text-anchor="middle" font-size="13" fill="#111">'
           f'mario-rl rollout profile &#183; Ryzen 7 7800X3D (8 cores / 16 threads)</text>']

    def axes(x0: float, title: str, top: float, unit: str) -> None:
        out.append(f'<text x="{x0}" y="{PAD_T - 12}" font-size="12" fill="#333">{title}</text>')
        out.append(f'<line x1="{x0}" y1="{PAD_T}" x2="{x0}" y2="{PAD_T + plot_h}" stroke="#ccc"/>')
        out.append(f'<line x1="{x0}" y1="{PAD_T + plot_h}" x2="{x0 + plot_w}" y2="{PAD_T + plot_h}" stroke="#ccc"/>')
        for i in range(5):
            v = top * i / 4
            y = PAD_T + plot_h - plot_h * i / 4
            out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" stroke="#eee"/>')
            out.append(f'<text x="{x0 - 6}" y="{y + 4:.1f}" text-anchor="end" fill="#666">{v:.0f}</text>')
        out.append(f'<text x="{x0 - 44}" y="{PAD_T + plot_h / 2}" fill="#666" font-size="10" '
                   f'transform="rotate(-90 {x0 - 44} {PAD_T + plot_h / 2})" text-anchor="middle">{unit}</text>')

    # left: stacked cost per vec step
    x0 = PAD_L
    axes(x0, "Where a step goes", y_max, "ms per vec step")
    for i, (envs, parts) in enumerate(zip(n, stacks)):
        cx = x0 + plot_w * (i + 0.5) / len(n)
        y = PAD_T + plot_h
        for (key, _, colour), val in zip(PARTS, parts):
            h = plot_h * val / y_max
            y -= h
            if h > 0.4:
                out.append(f'<rect x="{cx - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" '
                           f'fill="{colour}"/>')
        out.append(f'<text x="{cx:.1f}" y="{PAD_T + plot_h + 15}" text-anchor="middle" fill="#666">{envs}</text>')
    out.append(f'<text x="{x0 + plot_w/2:.1f}" y="{H - 20}" text-anchor="middle" fill="#666">'
               f'parallel environments</text>')

    # right: throughput
    x1 = PAD_L + plot_w + 72
    axes(x1, "Rollout throughput (no learning phase)", s_max, "agent steps / s")
    pts = [(x1 + plot_w * (i + 0.5) / len(n), PAD_T + plot_h - plot_h * v / s_max) for i, v in enumerate(sps)]
    ideal = [(x1 + plot_w * (i + 0.5) / len(n), PAD_T + plot_h - plot_h * (sps[0] * e) / s_max)
             for i, e in enumerate(n)]
    out.append('<polyline points="' + " ".join(f"{x:.1f},{max(y, PAD_T):.1f}" for x, y in ideal) +
               '" fill="none" stroke="#bbb" stroke-dasharray="4 3"/>')
    out.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
               '" fill="none" stroke="#4C78A8" stroke-width="2"/>')
    for (x, y), envs, v in zip(pts, n, sps):
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#4C78A8"/>')
        out.append(f'<text x="{x:.1f}" y="{PAD_T + plot_h + 15}" text-anchor="middle" fill="#666">{envs}</text>')
        out.append(f'<text x="{x:.1f}" y="{y - 8:.1f}" text-anchor="middle" fill="#4C78A8" '
                   f'font-size="9">{v:.0f}</text>')
    out.append(f'<text x="{x1 + plot_w - 4:.1f}" y="{PAD_T + 14}" text-anchor="end" fill="#999" '
               f'font-size="9">dashed = linear scaling from 1 env</text>')
    out.append(f'<text x="{x1 + plot_w/2:.1f}" y="{H - 20}" text-anchor="middle" fill="#666">'
               f'parallel environments</text>')

    # legend
    lx = PAD_L
    for _, label, colour in PARTS:
        out.append(f'<rect x="{lx}" y="{H - 40}" width="9" height="9" fill="{colour}"/>')
        out.append(f'<text x="{lx + 13}" y="{H - 32}" fill="#444">{label}</text>')
        lx += 20 + len(label) * 6.1
    out.append("</svg>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
