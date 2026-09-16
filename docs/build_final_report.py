"""Build docs/mario_rl_final_report.pdf from runs/dqn_1_1 (episodes.csv + eval JSONs).

    uv run --no-project --with reportlab --with matplotlib python docs/build_final_report.py
"""
import csv
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "dqn_1_1"
OUT = ROOT / "docs" / "mario_rl_final_report.pdf"
REPO = "https://github.com/NickNojiri/mario-rl"

# ---------------------------------------------------------------- data
rows = list(csv.DictReader(open(RUN / "episodes.csv")))
BUCKET = 250_000
buckets = {}
for r in rows:
    k = int(r["step"]) // BUCKET
    if k * BUCKET >= 3_000_000:  # single trailing episode past the end
        continue
    buckets.setdefault(k, []).append(r)
train = []
for k in sorted(buckets):
    g = buckets[k]
    n = len(g)
    train.append({
        "start": k * BUCKET, "end": (k + 1) * BUCKET, "episodes": n,
        "eps": sum(float(r["epsilon"]) for r in g) / n,
        "x": sum(int(r["x_pos"]) for r in g) / n,
        "past_pipes": sum(int(r["x_pos"]) > 722 for r in g) / n,
        "flags": sum(int(r["flag_get"]) for r in g),
        "reward": sum(float(r["reward"]) for r in g) / n,
    })
total_flags = sum(int(r["flag_get"]) for r in rows)
total_episodes = len(rows)


def load(name):
    return json.loads((RUN / name).read_text())


ev = {
    "rand11": load("eval_random_w1s1.json"), "dqn11": load("eval_step3000000_w1s1.json"),
    "rand12": load("eval_random_w1s2.json"), "dqn12": load("eval_step3000000_w1s2.json"),
    "greedy": load("eval_greedy_nonoop.json"),
}


def se(e):
    return e["std_reward"] / math.sqrt(e["episodes"])


def z_reward(a, b):
    return (a["mean_reward"] - b["mean_reward"]) / math.sqrt(se(a) ** 2 + se(b) ** 2)


z11, z12 = z_reward(ev["dqn11"], ev["rand11"]), z_reward(ev["dqn12"], ev["rand12"])
ratio11 = ev["dqn11"]["mean_x_pos"] / ev["rand11"]["mean_x_pos"]
ratio12 = ev["dqn12"]["mean_x_pos"] / ev["rand12"]["mean_x_pos"]

# ---------------------------------------------------------------- charts (validated: #2a78d6 / #eb6834, light)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def training_figure():
    mids = [(t["start"] + t["end"]) / 2 / 1e6 for t in train]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    fig.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0], [t["past_pipes"] * 100 for t in train], "Episodes past the 1-1 pipes (%)", "{:.0f}%"),
        (axes[1], [t["flags"] for t in train], "Flags reached per 250k steps", "{:.0f}"),
    ]
    for ax, ys, title, fmt in panels:
        style(ax)
        ax.axvspan(0, 1.0, color="#f0efec", zorder=0)
        ax.text(0.5, 0.97, "epsilon 1.0 -> 0.1", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7, color=MUTED)
        ax.plot(mids, ys, color=BLUE, linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        ax.set_title(title, loc="left", fontsize=9, color=INK, pad=8)
        ax.set_xlabel("Training steps (millions)")
        ax.set_xlim(0, 3.0)
        ax.set_ylim(bottom=0)
        for i in (0, len(ys) - 1):
            ax.annotate(fmt.format(ys[i]), (mids[i], ys[i]), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(w_pad=3)
    return png(fig)


def eval_figure():
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    groups = ["Level 1-1 (trained on)", "Level 1-2 (never seen)"]
    rand = [ev["rand11"]["mean_x_pos"], ev["rand12"]["mean_x_pos"]]
    dqn = [ev["dqn11"]["mean_x_pos"], ev["dqn12"]["mean_x_pos"]]
    w, gap = 0.34, 0.02
    xs = range(len(groups))
    b1 = ax.bar([x - w / 2 - gap for x in xs], rand, w, color=ORANGE, label="Random policy")
    b2 = ax.bar([x + w / 2 + gap for x in xs], dqn, w, color=BLUE, label="Trained DQN (3M steps)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30, f"{b.get_height():.0f}", ha="center",
                    fontsize=7.5, color=INK2)
    ax.set_xticks(list(xs), groups, color=INK2)
    ax.set_ylabel("Mean distance (x_pos)")
    ax.set_title("How far Mario gets: 30 eval episodes each, epsilon 0.01", loc="left", fontsize=9, pad=8)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(0, max(dqn + rand) * 1.2)
    fig.tight_layout()
    return png(fig)


# ---------------------------------------------------------------- document styles
INKc, MUTEDc, ACCENT, RULE, SHADE = (colors.HexColor(h) for h in ("#1f2328", "#57606a", "#1c5cab", "#d0d7de", "#f6f8fa"))
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=INKc,
                       alignment=TA_LEFT, spaceAfter=6)
META = ParagraphStyle("M", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=MUTEDc, spaceAfter=12)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INKc,
                    spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13.5,
                    textColor=ACCENT, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("B", parent=ss["Normal"], fontName="Helvetica", fontSize=9.8, leading=13.8, textColor=INKc,
                      spaceAfter=6)
CAP = ParagraphStyle("C", parent=BODY, fontSize=8.5, leading=11.5, textColor=MUTEDc, spaceAfter=10)
CELL = ParagraphStyle("CE", parent=BODY, fontSize=8.7, leading=11.4, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("BU", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
BOX = ParagraphStyle("BX", parent=BODY, backColor=SHADE, borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=6,
                     spaceAfter=12)
CODE = ParagraphStyle("CD", fontName="Courier", fontSize=8.2, leading=10.6, textColor=INKc, backColor=SHADE,
                      borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)
W = letter[0] - 1.5 * inch


def p(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(data, widths, header=True, align_right=()):
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
    c.drawString(0.75 * inch, 0.5 * inch, "Deep Q-Learning for Super Mario Bros  |  Nick Nojiri  |  September 2026")
    c.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, str(d.page))
    c.restoreState()


pct = lambda v: f"{v * 100:.0f}%"
s = []

# ---------------------------------------------------------------- title + summary
s += [
    p("Teaching an Agent to Play Super Mario Bros: Deep Q-Learning, Results, and Learning vs. Memorization", TITLE),
    p(f"Nick Nojiri &nbsp;|&nbsp; September 16, 2026 &nbsp;|&nbsp; Code: <link href='{REPO}' color='#1c5cab'>{REPO.replace('https://', '')}</link>", META),
]
s.append(p("Summary", H2))
s.append(p(
    f"I trained a Double Deep Q-Network (DQN) to play level 1-1 of Super Mario Bros from raw screen pixels, using only "
    f"a CPU. After 3 million decisions (about 9.7 hours), the agent travels <b>{ev['dqn11']['mean_x_pos']:.0f}</b> "
    f"units on average in a controlled evaluation, <b>{ratio11:.1f}x</b> farther than a random policy "
    f"({ev['rand11']['mean_x_pos']:.0f}). It reached the flag {total_flags} times during training. The evidence points to "
    f"a policy that reacts to the screen rather than replaying a memorized button sequence: it keeps that lead across "
    f"randomized start times that shift every enemy's position. It did <b>not</b> learn a general Mario skill: on "
    f"level 1-2, which it never trained on, it performs the same as random "
    f"({ev['dqn12']['mean_x_pos']:.0f} vs {ev['rand12']['mean_x_pos']:.0f}). Half of the evaluation episodes on 1-1 "
    f"stalled. In every stalled episode inspected, the jump button was being held, which points to a "
    f"partial-observability problem: the screen does not show that the button is already held.", BODY))

# ---------------------------------------------------------------- 1. setup
s.append(p("1. The learning problem", H2))
s.append(p("Reinforcement learning (RL) is learning by trial and error. The agent repeatedly observes the game, picks "
           "a button, and receives a reward. Its goal is a strategy (a <i>policy</i>) that maximizes total future "
           "reward. Formally this is a Markov Decision Process:", BODY))
s.append(table([
    ["Concept", "Plain meaning", "In this project"],
    ["State", "What the agent observes", "Last 4 frames, grayscale, 84x84 pixels (uint8)"],
    ["Action", "A button choice", "5 options: no-op, right, right+jump, right+run, right+jump+run"],
    ["Reward", "Score for the last move", "Rightward progress, minus a clock penalty, minus 15 for dying"],
    ["Episode", "One attempt", "Start of 1-1 until death, flag, or a no-progress cutoff"],
    ["Policy", "The learned strategy", "Pick the button with the highest predicted Q-value"],
], [0.9 * inch, 1.9 * inch, W - 2.8 * inch]))

# ---------------------------------------------------------------- 2. topics
s.append(p("2. ML and RL topics applied", H2))
s.append(p("Each technique below is in the codebase. For each: what it is, and why this project needed it.", BODY))
s.append(table([
    ["Topic", "What it is, simply", "How it was used / why"],
    ["Q-learning", "Q(state, action) = expected total future reward for pressing a button now",
     "The agent picks the highest-Q button; learning means making Q accurate."],
    ["Temporal-difference (TD) learning", "Improve a guess using a slightly better guess one step later",
     "Target = r + gamma * (1 - terminated) * Q_target(s', a'). No need to wait for an episode to end."],
    ["Discount factor (gamma)", "Future reward counts a little less than immediate reward",
     "gamma = 0.99 looks ~100 steps ahead. The tutorial's 0.9 looked ~10 steps, too short to plan a jump."],
    ["Terminated vs. truncated", "A true ending vs. an episode we cut short",
     "Death and flag zero out the future. The no-progress cutoff keeps bootstrapping, because the game would continue."],
    ["Convolutional neural network", "Layers that detect edges, then shapes, then objects",
     "Nature-DQN CNN (3 conv + 2 dense) maps 4 frames to 5 Q-values. It is never told what a goomba is."],
    ["Frame stacking", "Several frames in a row expose motion", "One frame cannot show whether Mario is rising or falling."],
    ["Experience replay", "Store past moves; train on random mini-batches",
     "Consecutive frames are highly correlated. Shuffled batches of 32 from a 100k buffer stabilize learning."],
    ["Target network", "A frozen copy of the network computes targets",
     "Synced every 10k steps so the goal does not move with every update."],
    ["Double DQN", "One network picks the next action, the other scores it", "Reduces the overestimation caused by always taking a max."],
    ["Huber loss + gradient clipping", "Limit how much one surprising sample can change the weights",
     "Prevents rare large TD errors from destabilizing training (grad norm capped at 10)."],
    ["Reward scaling", "Rescale rewards to a range the loss handles well",
     "Summed rewards reach +/-60 per step; dividing by 15 keeps TD errors in Huber's useful range."],
    ["Adam optimizer", "Gradient descent with a per-weight adaptive step size",
     "Its internal statistics are saved in checkpoints, so resumed training behaves identically."],
    ["Epsilon-greedy exploration", "Act randomly with probability epsilon",
     "Linear decay 1.0 -> 0.1 over 1M steps. The tutorial's multiplicative decay needed ~9.2M steps."],
    ["Generalization vs. overfitting", "Skill that transfers vs. memorizing the training data",
     "Measured with randomized evaluation and a held-out level (Sections 4-5)."],
    ["Partial observability / Markov property", "The state must contain everything needed to decide",
     "The main failure mode found: frames do not show that the jump button is held (Section 6)."],
    ["Baselines and statistical comparison", "Compare against chance, with uncertainty",
     "Random-policy evaluation plus standard errors over 30 episodes."],
], [1.45 * inch, 2.15 * inch, W - 3.6 * inch]))

s.append(p("Engineering that made the experiment possible", H3))
s += bullets([
    "<b>Memory:</b> storing each 84x84 frame once and rebuilding 4-frame stacks at sample time cut the replay buffer "
    "from 22.6 GB (float32 pairs) to <b>706 MB</b> with identical information.",
    "<b>Correctness:</b> 14 automated tests. The buffer is checked sample-by-sample against naive storage, and a resumed "
    "agent must produce an identical next update. One test caught a real bug: the no-progress counter started from "
    "the wrong position.",
    "<b>Reproducibility:</b> fixed seeds, pinned dependencies, config hash <font face='Courier'>8dd876e13de1</font>, "
    "and checkpoints that restore weights, optimizer, step count and every random-number state.",
    "<b>Hardware:</b> CPU-only (AMD Ryzen 7 7800X3D) under WSL Ubuntu, because the GPU is not CUDA-capable.",
])

# ---------------------------------------------------------------- 3. training
s.append(PageBreak())
s.append(p("3. Training run", H2))
s.append(table([
    ["Setting", "Value", "Setting", "Value"],
    ["Total agent steps", "3,000,000 (12M game frames)", "Replay buffer", "100,000 transitions"],
    ["Frame skip", "4", "Batch size / learn every", "32 / 3 steps"],
    ["Learning rate (Adam)", "2.5e-4", "Target sync", "every 10,000 steps"],
    ["gamma", "0.99", "Epsilon schedule", "1.0 -> 0.1 over 1M steps"],
    ["Reward scale", "1/15", "No-progress cutoff", "150 steps without new max x_pos"],
    ["Episodes", f"{total_episodes:,}", "Wall-clock time", "~9.7 hours (~85-125 steps/s)"],
], [1.35 * inch, 1.95 * inch, 1.5 * inch, W - 4.8 * inch]))
s.append(Image(training_figure(), width=W, height=W * 2.45 / 7.0))
s.append(p("<b>Figure 1.</b> Training progress in 250k-step windows. The shaded region is the exploration phase, where "
           "epsilon falls from 1.0 to 0.1. Around x_pos 722 sit the tall pipes that random play almost never clears.", CAP))
s.append(table(
    [["Steps", "Epsilon", "Episodes", "Mean x_pos", "Past pipes", "Flags"]] +
    [[f"{t['start'] / 1e6:.2f}M-{t['end'] / 1e6:.2f}M", f"{t['eps']:.2f}", f"{t['episodes']:,}", f"{t['x']:.0f}",
      pct(t["past_pipes"]), str(t["flags"])] for t in train],
    [1.35 * inch, 0.85 * inch, 0.9 * inch, 1.0 * inch, 1.0 * inch, W - 5.1 * inch]))
s.append(p("<b>Reading the curve.</b> Progress is clear through 1.75M steps (83% of episodes past the pipes). "
           "Performance then dips to 58% around 2.25-2.5M before recovering to 80%, while flags keep rising. Temporary "
           "regressions like this are a known property of DQN, which bootstraps from its own estimates. Because only the "
           "final checkpoint was kept, the 1.5-1.75M peak cannot be evaluated separately. The code now saves a snapshot "
           "every 250k steps.", BODY))

# ---------------------------------------------------------------- 4. evaluation
s.append(p("4. Evaluation protocol and results", H2))
s.append(p("Training scores include 10% random actions, so they understate and blur real skill. A separate, fixed "
           "evaluation measures the policy directly:", BODY))
s += bullets([
    "<b>30 episodes</b> per condition, <b>epsilon 0.01</b>, and a distinct seed per episode (reproducible, but not identical).",
    "<b>Random start delay:</b> 0-30 no-op steps before control begins, which shifts enemy positions relative to Mario.",
    "<b>Held-out level:</b> the same agent evaluated on level 1-2, which it never trained on.",
    "<b>Random-policy baseline</b> under the identical protocol, so every number can be compared with chance.",
    "<b>Unique-trajectory count:</b> how many of the 30 episodes used a different button sequence.",
])
s.append(table([
    ["Condition", "Mean x_pos", "Mean reward (+/- SE)", "Flag rate", "Stalled (cutoff)", "Unique runs"],
    ["1-1, random policy", f"{ev['rand11']['mean_x_pos']:.0f}", f"{ev['rand11']['mean_reward']:.0f} +/- {se(ev['rand11']):.0f}",
     pct(ev["rand11"]["flag_rate"]), pct(ev["rand11"]["no_progress_truncation_rate"]), "-"],
    ["<b>1-1, trained DQN</b>", f"<b>{ev['dqn11']['mean_x_pos']:.0f}</b>",
     f"<b>{ev['dqn11']['mean_reward']:.0f} +/- {se(ev['dqn11']):.0f}</b>", pct(ev["dqn11"]["flag_rate"]),
     pct(ev["dqn11"]["no_progress_truncation_rate"]), f"{ev['dqn11']['unique_trajectories']}/30"],
    ["1-2, random policy", f"{ev['rand12']['mean_x_pos']:.0f}", f"{ev['rand12']['mean_reward']:.0f} +/- {se(ev['rand12']):.0f}",
     pct(ev["rand12"]["flag_rate"]), pct(ev["rand12"]["no_progress_truncation_rate"]), "-"],
    ["<b>1-2, trained DQN (held out)</b>", f"<b>{ev['dqn12']['mean_x_pos']:.0f}</b>",
     f"<b>{ev['dqn12']['mean_reward']:.0f} +/- {se(ev['dqn12']):.0f}</b>", pct(ev["dqn12"]["flag_rate"]),
     pct(ev["dqn12"]["no_progress_truncation_rate"]), f"{ev['dqn12']['unique_trajectories']}/30"],
    ["1-1, trained DQN, fully greedy, no random start", f"{ev['greedy']['mean_x_pos']:.0f}",
     f"{ev['greedy']['mean_reward']:.0f} +/- 0", pct(ev["greedy"]["flag_rate"]), pct(ev["greedy"]["no_progress_truncation_rate"]),
     f"{ev['greedy']['unique_trajectories']}/30"],
], [1.95 * inch, 0.8 * inch, 1.3 * inch, 0.7 * inch, 0.95 * inch, W - 5.7 * inch]))
s.append(KeepTogether([
    Image(eval_figure(), width=5.2 * inch, height=2.5 * inch),
    p(f"<b>Figure 2.</b> Trained agent vs. random policy. On 1-1 the reward difference is {z11:.1f} standard errors; on "
      f"held-out 1-2 it is {z12:.1f}, which is indistinguishable from chance.", CAP),
]))

# ---------------------------------------------------------------- 5. learn vs memorize
s.append(PageBreak())
s.append(p("5. Did it learn, or did it memorize?", H2))
s.append(p("This is the central question for any agent trained on a single, deterministic level. The NES emulator "
           "has no randomness: the same buttons always produce the same outcome. An agent could therefore score well by "
           "replaying one memorized button sequence, like memorizing exam answers instead of learning the material. "
           "The project was designed to detect and discourage that.", BODY))

s.append(p("How the setup pushes toward learning", H3))
s.append(table([
    ["Design choice", "Why it discourages memorization"],
    ["Random 0-30 step start delay (training and eval)",
     "Enemies are in different places relative to Mario each episode, so a fixed tape of button presses breaks. The "
     "agent must react to what is on screen."],
    ["Epsilon-greedy noise during training (10% after 1M steps)",
     "Random actions knock the agent off any single path, so it must recover from many different situations."],
    ["Experience replay over 100k transitions",
     "Each update mixes moments from thousands of episodes rather than rehearsing the latest run."],
    ["Pixel input only; no map, route or positions supplied",
     "Nothing tells the agent where to go. It only sees the screen and the score."],
    ["Evaluation with distinct seeds and random start offsets",
     "Scores come from many different situations, so a policy that only works along one path cannot score well."],
    ["Held-out level 1-2 and a random baseline", "Separates skill on 1-1 from general skill, and both from chance."],
], [2.3 * inch, W - 2.3 * inch]))

s.append(p("What the evidence shows", H3))
s.append(table([
    ["Evidence", "Result", "Interpretation"],
    ["Trained vs. random on 1-1, across 31 random start offsets", f"{ratio11:.1f}x farther; reward +{z11:.1f} SE",
     "Genuine, statistically clear learning. A fixed memorized sequence would fall apart once enemies shift; this "
     "policy keeps its lead, so it is reacting to the screen."],
    ["Distinct trajectories on 1-1", f"{ev['dqn11']['unique_trajectories']} of 30",
     "Consistent with the above, but weak on its own: different start delays already change the sequence."],
    ["Trained vs. random on held-out 1-2", f"{ratio12:.2f}x; reward +{z12:.1f} SE",
     "<b>No transfer.</b> The skill is specific to 1-1's layout and look."],
    ["Fully greedy, fixed start", "1 trajectory, dies at x_pos 898 every time",
     "With all randomness removed, the policy is deterministic and brittle at one spot."],
], [2.05 * inch, 1.55 * inch, W - 3.6 * inch]))
s.append(p("<b>Conclusion.</b> The agent <b>learned a reactive policy for level 1-1</b> rather than memorizing a button "
           "sequence: it keeps a large, statistically clear lead over chance even when every episode starts with enemies "
           "in different places. "
           "It did <b>not</b> learn to play Mario in general, because it gained nothing on an unseen level. That is the "
           "expected outcome of training on a single level: the network only ever saw 1-1, so the features it learned "
           "(this pipe, these gaps, this background) do not describe 1-2's underground visuals. Generalization "
           "has to be trained for, most directly by training on several levels at once.", BOX))

# ---------------------------------------------------------------- 6. failure analysis
s.append(p("6. Failure analysis: why half the evaluation episodes stalled", H2))
s.append(p("Fifty percent of trained 1-1 evaluation episodes ended through the no-progress cutoff rather than death or "
           "the flag. A per-episode diagnosis of 12 episodes found 1 flag, 7 deaths and 4 stalls. <b>In all 4 stalls, "
           "the jump button (A) was held for 100% of the final 150 steps.</b>", BODY))
s.append(p("In Super Mario Bros, A must be released before Mario can jump again. The agent's state is only screen "
           "pixels, and pixels do not show whether A is currently held. Two situations that look identical on screen "
           "need different actions (\"press A\" vs. \"release A first\"), so the state violates the <b>Markov "
           "property</b>. This is <b>partial observability</b>. Facing a pipe or staircase, the highest-Q action is "
           "\"jump\", but holding jump does nothing, and the agent has no way to perceive why.", BODY))
s.append(p("During training, the 10% random actions released A by chance, which hid the problem. At evaluation "
           "epsilon 0.01, almost nothing does. The planned fix is to feed the previous action into the network "
           "alongside the frames, making the button state observable.", BODY))

# ---------------------------------------------------------------- 7. next + repro
s.append(p("7. Limitations and next steps", H2))
s += bullets([
    "<b>Single training level:</b> train on several stages (e.g. 1-1 to 1-4) and hold out others to measure real generalization.",
    "<b>Partial observability:</b> add the previous action as a network input; compare against a recurrent (LSTM) variant.",
    "<b>One seed, one run:</b> RL results vary across seeds; repeat with 3+ seeds and report the spread.",
    "<b>Checkpoint selection:</b> evaluate the new 250k-step snapshots to measure the mid-run dip and pick the best policy.",
    "<b>Stochasticity:</b> add sticky actions (repeat the last action with small probability) as a further guard against memorization.",
])

s.append(p("8. Reproducibility", H2))
s.append(p(f"All code, tests and this report's generator are public at <link href='{REPO}' color='#1c5cab'>{REPO}</link>. "
           "Run 1 used commit <font face='Courier'>90081ba</font>; results and figures come directly from the run's logs.", BODY))
s.append(Preformatted(
    "python train.py --preset full --run-name dqn_1_1        # training (3M steps)\n"
    "python eval.py runs/dqn_1_1/latest.pt --episodes 30     # evaluation on 1-1\n"
    "python eval.py runs/dqn_1_1/latest.pt --stage 2         # held-out level 1-2\n"
    "python eval.py runs/dqn_1_1/latest.pt --epsilon 1.0     # random-policy baseline\n"
    "python -m pytest tests                                   # 14 tests", CODE))
s.append(p("<b>Acknowledgement.</b> Software development for this project was done with an AI coding assistant "
           "(Claude Code). Experiments were run locally, and all reported numbers come from those runs.", CAP))

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.75 * inch, bottomMargin=0.8 * inch, author="Nick Nojiri",
                        title="Deep Q-Learning for Super Mario Bros: Results and Learning vs. Memorization")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
print(f"z11={z11:.2f} z12={z12:.2f} ratio11={ratio11:.2f} ratio12={ratio12:.2f}")
