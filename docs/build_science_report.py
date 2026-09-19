"""Build docs/mario_rl_science.pdf: what works, the method, every experiment and what is still open.

    uv run --no-project --with reportlab --with matplotlib python docs/build_science_report.py
"""
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Table,
                                TableStyle)

OUT = Path(__file__).resolve().parent / "mario_rl_science.pdf"
REPO = "https://github.com/NickNojiri/mario-rl"

# ---------------------------------------------------------------- results (all from runs/ eval JSONs and logs)
RUNS = [  # label, held-out x_pos, training x_pos, training flag rate, note
    ("Random policy", 382, 443, 0.0, "chance"),
    ("v2: tiles, game reward", 582, 774, 0.005, "1.72M steps"),
    ("v3b: new reward", 648, 924, 0.018, "1.72M steps, best short run"),
    ("v3c: + enemy motion", 567, 860, 0.014, "1.72M steps"),
    ("v3d: + 12 buttons", 560, 853, 0.005, "1.72M steps"),
    ("v4: macros, hints, modes", 576, 830, 0.032, "1.72M steps"),
    ("v3b, 4.5x longer", 674, 1440, 0.118, "7.8M steps, best overall"),
    ("gen1: 75% generated", 446, 757, 0.050, "7.8M steps"),
    ("gen2: 40% generated, harder", 517, 1135, 0.123, "7.8M steps"),
]
GEN2_CURVE = [(1.5, 449), (3.0, 499), (4.5, 549), (6.0, 502), (7.5, 591)]
SWEEP = [("macros", 479, 34), ("points x3", 459, 30), ("more exploration", 455, 43), ("lower learn rate", 454, 30),
         ("bigger batch", 443, 25), ("base", 433, 29), ("physics hints", 416, 27), ("prioritized stages", 413, 29),
         ("enemy motion", 398, 21), ("stricter stall cutoff", 389, 18), ("death -400", 386, 18)]

BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#eb6834", "#898781", "#1baf7a"
SURFACE, INK_C, GRID_C, AXIS_C = "#fcfcfb", "#0b0b0b", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS_C,
                     "text.color": INK_C, "xtick.color": "#57606a", "ytick.color": "#57606a"})


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def fig_runs():
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    labels = [r[0] for r in RUNS]
    vals = [r[1] for r in RUNS]
    cols = [GREY] + [BLUE] * 5 + [GREEN] + [ORANGE] * 2
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:.0f}", ha="center", fontsize=7.5, color="#57606a")
    ax.axhline(382, color=GREY, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(vals) - 0.4, 392, "random", fontsize=7, color=GREY, ha="right")
    ax.set_xticks(range(len(labels)), labels, rotation=18, ha="right", fontsize=7.5)
    ax.set_ylabel("Held-out distance (x_pos)")
    ax.set_title("Every experiment, measured the same way: 5 levels never trained on", loc="left", fontsize=9.5, pad=8)
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    return png(fig)


def fig_curve_and_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    style(ax)
    ax.plot([x for x, _ in GEN2_CURVE], [y for _, y in GEN2_CURVE], color=BLUE, linewidth=2, marker="o",
            markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(674, color=GREEN, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(7.4, 686, "v3b (real levels only)", fontsize=7, color=GREEN, ha="right")
    ax.axhline(382, color=GREY, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(7.4, 394, "random", fontsize=7, color=GREY, ha="right")
    ax.set_title("gen2: held-out check every 1.5M steps", loc="left", fontsize=9, pad=8)
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Held-out x_pos")
    ax.set_ylim(300, 760)

    ax = axes[1]
    style(ax)
    names = [s[0] for s in SWEEP][::-1]
    vals = [s[1] for s in SWEEP][::-1]
    errs = [s[2] for s in SWEEP][::-1]
    cols = [ORANGE if n == "base" else BLUE for n in names]
    ax.barh(range(len(vals)), vals, xerr=errs, color=cols, height=0.6,
            error_kw={"ecolor": "#57606a", "elinewidth": 1, "capsize": 2})
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Held-out x_pos (+- standard error)")
    ax.set_title("Parameter sweep: no change beat the base", loc="left", fontsize=9, pad=8)
    ax.set_xlim(300, 540)
    fig.tight_layout(w_pad=3)
    return png(fig)


# ---------------------------------------------------------------- document
INKc, MUTEDc, ACCENT, RULE, SHADE = (colors.HexColor(h) for h in ("#1f2328", "#57606a", "#1c5cab", "#d0d7de", "#f6f8fa"))
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=INKc,
                       alignment=TA_LEFT, spaceAfter=6)
META = ParagraphStyle("M", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=MUTEDc, spaceAfter=12)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INKc,
                    spaceBefore=13, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13.5,
                    textColor=ACCENT, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("B", parent=ss["Normal"], fontName="Helvetica", fontSize=9.7, leading=13.6, textColor=INKc,
                      spaceAfter=6)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTEDc, spaceAfter=10)
CELL = ParagraphStyle("CE", parent=BODY, fontSize=8.6, leading=11.3, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("BU", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
BOX = ParagraphStyle("BX", parent=BODY, backColor=SHADE, borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=6,
                     spaceAfter=12)
CODE = ParagraphStyle("CD", fontName="Courier", fontSize=8.1, leading=10.5, textColor=INKc, backColor=SHADE,
                      borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)
W = letter[0] - 1.5 * inch


def p(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(data, widths, header=True):
    cells = [[Paragraph(str(c), CELLB if header and i == 0 else CELL) for c in r] for i, r in enumerate(data)]
    t = Table(cells, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), SHADE), ("LINEBELOW", (0, 0), (-1, 0), 0.9, INKc)]
    t.setStyle(TableStyle(st))
    t.spaceAfter = 9
    return t


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTEDc)
    c.drawString(0.75 * inch, 0.5 * inch, "mario-rl  |  what works and how we test it  |  September 19, 2026")
    c.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, str(d.page))
    c.restoreState()


s = [p("Teaching Mario: What Works, and How We Test It", TITLE),
     p(f"Reinforcement learning on real Super Mario Bros, CPU only. Status report and method. "
       f"<link href='{REPO}' color='#1c5cab'>{REPO.replace('https://', '')}</link>", META)]

s.append(p("1. Where the project stands", H2))
s.append(table([
    ["Piece", "State", "Evidence"],
    ["Environment and tooling (WSL, pinned deps, 57 tests)", "<b>Working</b>",
     "Tests cover replay memory, advantages, observations, rewards, generated terrain"],
    ["Tile-grid observation read from game memory", "<b>Working</b>",
     "Verified against screenshots on overworld, underground and castle stages"],
    ["PPO trainer (12 parallel games, ~600 steps/s)", "<b>Working</b>",
     "Resumable, snapshots, CSV logs, detached launcher"],
    ["Evaluation protocol (held-out levels, random baseline)", "<b>Working</b>",
     "Same seeds and protocol for every run; uncertainty reported"],
    ["Procedural level generator inside the real game", "<b>Working</b>",
     "Terrain, pipes, tile themes verified solid; flags reachable; enemies kept on the surface"],
    ["Live viewers (single and multi-game, 1x-8x)", "<b>Working</b>", "Browser stream, follows training"],
    ["An agent that generalizes to unseen levels", "<b>Not yet</b>",
     "Best is 674 vs 382 random; no flags on held-out levels"],
], [2.45 * inch, 0.8 * inch, W - 3.25 * inch]))

s.append(p("2. The scientific approach", H2))
s.append(p("The project runs as a loop: state a claim, measure it, keep only what survives measurement. "
           "Six rules have done most of the work.", BODY))
s.append(table([
    ["Rule", "How it is applied here"],
    ["<b>1. One primary number, fixed in advance</b>",
     "Held-out distance: mean x_pos over 5 real levels never trained on, 10 episodes each, fixed seeds, random "
     "start delays. A random policy (382) anchors the bottom of the scale."],
    ["<b>2. Know the noise before comparing</b>",
     "Standard error is measured, not assumed: +-36 with 6 episodes per level, +-25 with 12. Differences under "
     "about 2 standard errors are reported as noise, not results."],
    ["<b>3. Verify every claim about the game</b>",
     "Nine probe scripts check memory layout and physics against the running emulator: timer expiry kills Mario, "
     "enemy speeds and positions, when level columns are written, and a measured jump table."],
    ["<b>4. Change one thing at a time, equal budget</b>",
     "Every variant trains from scratch for the same number of steps and gets the same evaluation. The 18-variant "
     "sweep and the v3/v4 experiments follow this."],
    ["<b>5. Keep the test set clean</b>",
     "Held-out levels are never trained on, never used to pick settings, and generated terrain can never appear "
     "in evaluation (any evaluation names its level explicitly, which disables generation)."],
    ["<b>6. Report negative results and diagnose them</b>",
     "Most ideas failed. Each failure gets a measurement: why it failed, and what it implies."],
], [1.85 * inch, W - 1.85 * inch]))

s.append(p("Instruments built along the way", H3))
s += bullets([
    "<b>Jump table</b> (scripted jumps): max distance 3.3 tiles standing, 5.2 walking, 9.8 running; a full jump "
    "needs A held 6-8 decisions. This defines what terrain is possible.",
    "<b>Pit forensics</b>: classifies every pit death (walked off, released A early, too slow, took off too early) "
    "and measures how long the danger was visible.",
    "<b>Behaviour logging</b>: cause of death, stomps, jumps, LEFT/NOOP usage and every reward component, per episode.",
    "<b>Early-warning evaluation</b>: held-out check at every snapshot, so a run that is going wrong shows it "
    "hours before it ends.",
])

s.append(p("3. Every experiment, same measurement", H2))
s.append(Image(fig_runs(), width=W, height=W * 2.6 / 7.0))
s.append(p("<b>Figure 1.</b> Held-out distance for each configuration. Blue: equal-length (1.72M step) runs. "
           "Green: the best overall (same recipe, 4.5x longer). Orange: generated-level runs.", CAP))
s.append(table(
    [["Run", "Held-out x_pos", "Training x_pos", "Training flag rate", "Note"]] +
    [[r[0], f"{r[1]}", f"{r[2]}", f"{r[3]:.1%}", r[4]] for r in RUNS],
    [1.9 * inch, 1.0 * inch, 1.0 * inch, 1.05 * inch, W - 4.95 * inch]))
s.append(Image(fig_curve_and_sweep(), width=W, height=W * 2.4 / 7.0))
s.append(p("<b>Figure 2.</b> Left: the generated-level run improves steadily on held-out levels and had not "
           "flattened when it stopped. Right: 18 one-change variants; every bar overlaps the base within noise.", CAP))

s.append(p("4. What has actually been learned", H2))
s.append(table([
    ["Finding", "Evidence", "Consequence"],
    ["Reward design mattered more than tuning",
     "New reward: 582 -> 648 held-out. 18 tuning variants: none beyond noise",
     "Stopped tuning; spent effort on the task instead"],
    ["More training improves known levels, not new ones",
     "4.5x longer: training flags 1.8% -> 11.8%, held-out 648 -> 674",
     "The gap is memorization, not lack of practice"],
    ["The agent could not see enemy motion",
     "A goomba moves 2 px per decision; 4 stacked 16 px grids hide direction",
     "Added exact enemy positions, speeds and states"],
    ["Half of pit deaths were walk-offs, not bad jumps",
     "Pit visible ~12 decisions ahead, yet jump probability at the edge was 48%; "
     "failed jumps held A 2 steps vs 5 for successes",
     "Added macro jumps (hold A for a fixed time): releasing A early fell from 14% to 1% of pit deaths"],
    ["Generated levels can be built inside the real game",
     "Terrain written into the collision map; pits, walls, pipes and themes verified solid; flags reachable",
     "Unlimited new layouts with the game's own physics"],
    ["Training mostly on generated levels backfired",
     "75% generated: 33% of generated levels finished but held-out fell to 446 and randomness collapsed",
     "Generated levels must resemble real ones, and stay a minority of practice"],
    ["Harder, more realistic generated levels recovered most of it",
     "40% generated, harder terrain, pipes, tile themes, faster enemies: held-out 517, and equal to the best run "
     "on 4 of 5 held-out levels; the castle level accounts for the gap",
     "Generated castles (lava, fire bars) are the next piece"],
], [1.8 * inch, 2.6 * inch, W - 4.4 * inch]))

s.append(p("5. What is still open", H2))
s += bullets([
    "<b>No held-out level has ever been finished.</b> The best agent gets about a fifth of the way through, so the "
    "honest claim is 'clearly better than chance, far from playing Mario'.",
    "<b>Castles are the weak spot</b> (held-out 5-4: 320 vs 948). Lava reads as an ordinary pit and fire-bar arms "
    "are invisible in the observation.",
    "<b>Single seed per configuration.</b> Reinforcement learning varies between runs; conclusions here rest on one "
    "run each, plus the measured evaluation noise.",
    "<b>Distance is a proxy.</b> Levels differ in length, so x_pos is comparable across runs on the same levels, "
    "not across levels.",
])

s.append(p("6. Running now, and next", H2))
s.append(p("The harder-generator run is continuing to 15M steps, with a held-out check every 2M. Next: generated "
           "castle levels, then a like-for-like rematch against the best real-levels-only run at equal steps.", BODY))
s.append(Preformatted(
    "python train_ppo.py --preset ppo_gen2 --run-name gen2     # 40% generated, harder, faster\n"
    "python eval_stages.py runs/gen2/latest.pt                 # held-out + training levels\n"
    "python -m scripts.sweep --stage a                         # one-change variants, ranked\n"
    "python -m scripts.pit_forensics runs/gen2/latest.pt       # why it dies\n"
    "python -m scripts.multi_demo --envs 4 --speed 2 --procgen 0.5   # watch 4 games at 2x", CODE))
s.append(p("<b>Reproducibility.</b> Pinned environment, fixed seeds, config hash in every checkpoint, 57 automated "
           "tests, and every figure here traceable to a run log or evaluation file in the repository. Software was "
           "developed with an AI coding assistant; all numbers come from runs on this machine.", CAP))

SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch,
                  bottomMargin=0.8 * inch, title="Teaching Mario: what works and how we test it",
                  author="Nick Nojiri").build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
