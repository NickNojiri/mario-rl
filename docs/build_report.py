from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = r"C:\Users\17143\mario-rl\docs\mario_rl_report.pdf"

INK = colors.HexColor("#1f2328")
MUTED = colors.HexColor("#57606a")
ACCENT = colors.HexColor("#c0392b")
RULE = colors.HexColor("#d0d7de")
SHADE = colors.HexColor("#f6f8fa")
OK = colors.HexColor("#1a7f37")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=22, leading=27, textColor=INK,
                    alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=14)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK,
                    spaceBefore=14, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ACCENT,
                    spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica", fontSize=9.8, leading=13.6, textColor=INK,
                      spaceAfter=6)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.8, leading=11.6, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=3)
CODE = ParagraphStyle("CODE", fontName="Courier", fontSize=8.3, leading=10.8, textColor=INK, backColor=SHADE,
                      borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=10)
CALLOUT = ParagraphStyle("CALLOUT", parent=BODY, backColor=SHADE, borderColor=ACCENT, borderWidth=0,
                         borderPadding=8, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=12)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="\u2022") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SHADE), ("LINEBELOW", (0, 0), (-1, 0), 0.9, INK)]
    t.setStyle(TableStyle(style))
    t.spaceAfter = 9
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.5 * inch, "mario-rl  |  status report  |  September 14, 2026")
    canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"{doc.page}")
    canvas.restoreState()


W = letter[0] - 1.5 * inch
s = []

# ---------------- Title ----------------
s += [
    p("Mario RL: What Works, What We Did, and Why", H1),
    p("Double-DQN agent for Super Mario Bros 1-1, built and training on a CPU-only Windows + WSL machine. "
      "Ends with design options for a Pokemon agent. Snapshot taken September 14, 2026, ~20 minutes into the full run.",
      SUB),
]

s.append(p("Status at a glance", H2))
s.append(table([
    ["Component", "State", "Evidence"],
    ["Environment (WSL, Python 3.10, pinned deps)", "<font color='#1a7f37'><b>Working</b></font>",
     "nes-py compiled with gcc; import probe passes; requirements.txt frozen"],
    ["Env adapter (env/make_env.py)", "<font color='#1a7f37'><b>Working</b></font>",
     "4 env tests pass against the real emulator"],
    ["Frame replay buffer (agent/buffer.py)", "<font color='#1a7f37'><b>Working</b></font>",
     "3 tests: every sample matches naive storage, incl. wraparound and 1-step episodes"],
    ["Double-DQN agent + checkpoints (agent/mario.py)", "<font color='#1a7f37'><b>Working</b></font>",
     "6 tests: resumed agent makes the identical next update and actions"],
    ["train.py / eval.py / config presets", "<font color='#1a7f37'><b>Working</b></font>",
     "Smoke run, resume with buffer, resume without buffer, eval all run end to end"],
    ["Detached launcher + clean stop", "<font color='#1a7f37'><b>Working</b></font>",
     "SIGTERM test wrote a final checkpoint; full run survives closing the terminal"],
    ["Full training run dqn_1_1", "<font color='#9a6700'><b>In progress</b></font>",
     "Step 136,735 of 3,000,000; epsilon 0.877; ~120 steps/s; ETA about 11:30 PM to midnight"],
    ["Trained agent quality", "<font color='#57606a'><b>Unknown yet</b></font>",
     "Needs eval.py once epsilon reaches 0.1 at step 1M"],
], [2.2 * inch, 0.95 * inch, W - 3.15 * inch]))

s.append(p("Test suite: 14 of 14 passing. Git: two local commits on <b>main</b>, nothing pushed.", BODY))

# ---------------- What we did ----------------
s.append(p("1. Reviewed the plan before building", H2))
s.append(p("The starting point was a seven-phase plan to fix the PyTorch Mario tutorial. Most of it was sound. "
           "Before writing code, each claim was checked against the gym-super-mario-bros source and, later, the running "
           "emulator. Four reasons were wrong even where the fix was right, and four things were missing.", BODY))
s.append(table([
    ["Plan claim", "What is actually true", "Outcome"],
    ["Mario's 400-tick clock causes truncation, so bootstrapping is broken (the top-priority fix).",
     "Timer expiry <b>kills Mario</b>: the probe ran NOOPs until time=0 and got terminated=True, truncated=False. "
     "gym's TimeLimit is registered at 9,999,999 steps, so truncated is never set.",
     "Kept the terminated/truncated split (cheap and correct) but demoted it. It only matters with our own cutoff."],
    ["Hardcoded batch_size in Q[np.arange(B), action] silently indexes wrong rows.",
     "Mismatched lengths usually <b>raise</b>. The silent bug is an action shaped [B,1], which broadcasts to a BxB matrix.",
     "Kept gather(), plus shape asserts and squeeze(-1) everywhere."],
    ["NumPy 2 breaks nes-py's C extension and segfaults.",
     "The confirmed break is in Python: smb_env computes uint8 x 256 for x_pos, which NumPy 2 rejects.",
     "Kept the numpy&lt;2 pin with the correct reason."],
    ["Restarting without Adam state wrecks a model.",
     "Bias correction makes it a jolt, not a wreck. The bigger resume hole was the <b>empty replay buffer</b>.",
     "Checkpoints save optimizer + RNG; learning waits for burn-in when the buffer is not restored."],
], [1.85 * inch, 2.75 * inch, W - 4.6 * inch]))

s.append(p("Missing from the plan, added:", H3))
s += bullets([
    "<b>Reward scaling.</b> Per-frame reward is clipped to +/-15; four skipped frames sum to +/-60. With Huber loss "
    "(beta=1) errors that large sit on the linear part, so every gradient has the same magnitude. Stored reward is "
    "divided by 15; logs and eval stay in raw units.",
    "<b>No-progress cutoff.</b> A Mario stuck on a pipe burns about 8,000 frames of useless data. Episodes are truncated "
    "if the best x_pos has not improved in 150 agent steps. This is a genuine truncation, where bootstrapping is correct.",
    "<b>Eval that detects memorization.</b> The NES emulator is deterministic, so a near-greedy policy replays the same "
    "run. Eval uses per-episode seeds, random NOOP starts, held-out stages, and counts unique action sequences.",
    "<b>A smoke preset that exercises everything.</b> The tutorial syncs the target every 10k steps and saves every 500k, "
    "so a short test never reaches those paths. Smoke syncs every 500 and saves every 1,000.",
])

s.append(p("2. Set up the environment", H2))
s.append(table([
    ["Finding", "Decision", "Why"],
    ["GPU is an AMD RX 5700 XT", "Train on CPU (Ryzen 7 7800X3D, 16 threads)",
     "PyTorch CUDA needs NVIDIA; ROCm does not support this card on Windows. The emulator is the bottleneck for a "
     "small CNN anyway."],
    ["nes-py ships only a source tarball for Windows; VS Build Tools had no C++ compiler",
     "Run everything in WSL Ubuntu 24.04", "gcc 13 was already installed; no admin rights or 6-8 GB download needed."],
    ["nes-py targets the old gym API and NumPy 1.x", "Python 3.10, gym 0.26.2, numpy 1.26.4, torch 2.14 CPU",
     "Locking the interpreter is cheaper than patching a C extension we do not own. Frozen right after the import test."],
    ["Linux venv on /mnt/c is slow", "Venv in ~/.venvs/mario-rl; code stays in C:\\Users\\17143\\mario-rl",
     "Code is editable from Windows; Python startup and imports stay fast."],
], [1.9 * inch, 1.9 * inch, W - 3.8 * inch]))

s.append(p("3. Built the project", H2))
s.append(Preformatted(
    "mario-rl/\n"
    "  config.py            dataclass presets (smoke, full) + hash of learning fields\n"
    "  env/make_env.py      the ONLY file that knows gym/nes-py quirks\n"
    "  agent/buffer.py      frame-deduplicated replay buffer\n"
    "  agent/net.py         Nature-DQN CNN; uint8 in, /255 inside forward()\n"
    "  agent/mario.py       act / cache / learn / save / load\n"
    "  train.py  eval.py    training loop with CSV log; fixed-protocol evaluation\n"
    "  scripts/             probe_env.py, start_train.ps1, stop_train.ps1, wsl_run.sh\n"
    "  tests/               buffer, agent, env (14 tests)", CODE))

s.append(p("Environment adapter", H3))
s.append(p("Everything outside <b>make_env.py</b> sees a Gymnasium-style API: reset(seed) returns (obs, info), step returns "
           "(obs, reward, terminated, truncated, info), and obs is uint8 [4, 84, 84]. Inside, it does frame skip "
           "(stopping on either flag), grayscale + INTER_AREA resize, frame stacking, seeded 0-30 NOOP starts, and the "
           "no-progress cutoff. Quirks it absorbs: nes-py's JoypadSpace.reset() rejects a seed argument, and gym's "
           "compatibility layer returns an empty info dict from reset(). A test caught that second one: it made the "
           "first step always look like progress.", BODY))

s.append(p("Replay buffer: store each frame once", H3))
s.append(table([
    ["Storage scheme", "Bytes per transition", "100k transitions"],
    ["float32 state + next_state (tutorial)", "4 x 84 x 84 x 4 bytes x 2 = 225,792", "22.6 GB (spills to disk memmap)"],
    ["uint8 state + next_state (original plan)", "4 x 84 x 84 x 2 = 56,448", "5.6 GB"],
    ["uint8, one frame per row (built)", "84 x 84 = 7,056", "<b>706 MB</b>"],
], [2.3 * inch, 2.4 * inch, W - 4.7 * inch]))
s.append(p("Consecutive stacks share three of four frames, so the buffer stores only each observation's newest frame and "
           "rebuilds stacks at sample time, repeating the first frame across episode starts exactly as reset() does. "
           "An episode's final observation gets its own row so a truncated transition still has a real next state. "
           "This also removed torchrl and tensordict, which must be version-matched to torch.", BODY))

s.append(p("Agent", H3))
s += bullets([
    "Double DQN: online network picks a' , target network scores it; both lookups use gather().",
    "TD target = r + gamma * (1 - terminated) * Q_target(s', a'). Truncation keeps the bootstrap.",
    "Linear epsilon 1.0 to 0.1 over 1M steps. The tutorial's 0.99999975 multiplier needs ~9.2M steps to reach 0.1.",
    "gamma 0.99 (tutorial 0.9 looks only ~10 agent steps, about 0.7 s of game time, ahead). Gradient norm clipped at 10.",
    "Checkpoint: weights, target, Adam state, step, episode, all RNG states, config, learning hash, and optionally the "
    "buffer. Resuming with a changed learning field is refused unless --allow-config-change is passed.",
])

s.append(p("4. Verified", H2))
s.append(table([
    ["Check", "Result"],
    ["Unit + integration tests", "14/14 pass (about 6 s)"],
    ["Smoke training, 2,000 steps", "~20 s; learning, 3 target syncs, 2 checkpoints; best smoke episode reached x_pos 1510"],
    ["Resume with saved buffer", "Continued at step 2,000 with eps and buffer intact"],
    ["Resume without buffer", "Refilled 200 transitions, then resumed learning"],
    ["Learning-field override on resume", "Refused with a clear message"],
    ["SIGTERM during training", "Caught; final checkpoint written"],
    ["eval.py on smoke checkpoint", "10 episodes, 10 unique trajectories, mean x_pos 625 (expected for 3k steps)"],
], [2.4 * inch, W - 2.4 * inch]))

sec5 = []
sec5.append(p("5. Training now", H2))
sec5.append(p("Run <b>dqn_1_1</b> started around 4:30 PM with the full preset: 3M steps, 100k buffer, burn-in 10k, checkpoint "
           "every 100k steps including the buffer. At step 136,735 the agent is still 88% random, so reward is mostly "
           "noise. Mean Q is rising slowly (about 3.3) with small, stable loss.", BODY))
sec5 += bullets([
    "<b>Watch x_pos.</b> Random play stalls around 700-722, the tall pipes of 1-1. Clearing them consistently is the "
    "first sign of real learning.",
    "<b>Sleep pauses training.</b> Keep the PC awake until it finishes.",
    "<b>First real measurement:</b> eval.py on the step-1M checkpoint, on 1-1 and on held-out 1-2.",
])
sec5.append(Preformatted(
    "# PowerShell: watch / stop + save / resume\n"
    "Get-Content C:\\Users\\17143\\mario-rl\\runs\\dqn_1_1\\stdout.log -Tail 5 -Wait\n"
    "C:\\Users\\17143\\mario-rl\\scripts\\stop_train.ps1\n"
    "cd C:\\Users\\17143\\mario-rl; .\\scripts\\start_train.ps1 -Resume runs/dqn_1_1/latest.pt\n"
    "\n"
    "# WSL venv: evaluate on 1-1, then on held-out 1-2\n"
    "python eval.py runs/dqn_1_1/latest.pt --episodes 30\n"
    "python eval.py runs/dqn_1_1/latest.pt --stage 2", CODE))
s.append(KeepTogether(sec5))

# ---------------- Pokemon ----------------
s.append(PageBreak())
s.append(p("Next: a Pokemon agent", H1))
s.append(p("Pokemon demands longer-horizon reasoning than Mario. This section covers what changes, the build options, "
           "and where to draw the line on giving the agent outside knowledge.", SUB))

s.append(p("Why it is harder", H2))
s.append(table([
    ["", "Mario 1-1", "Pokemon Red"],
    ["Reward density", "Every step: moving right pays", "Badges and new areas arrive thousands of steps apart"],
    ["Memory needed", "4 frames is enough", "Where have I been, what am I carrying, which events happened"],
    ["Episode end", "Death or flag (true terminals)", "No death; nearly every episode is truncated"],
    ["Reasoning", "Timing jumps", "Type matchups, switching, item and HM gating, route planning"],
    ["Action structure", "5 buttons, immediate effect", "Menus nested several levels deep"],
], [1.3 * inch, 2.2 * inch, W - 3.5 * inch]))

s.append(p("Build options", H2))
s.append(table([
    ["Option", "How", "Realistic on this PC"],
    ["<b>A. Pokemon Red: explore + battle</b>",
     "PyBoy Game Boy emulator (fast, headless, Python). PPO across ~12-16 parallel emulators. Reward read from RAM. "
     "Prior art: Peter Whidden's open-source PokemonRedExperiments.",
     "Leaving Pallet Town and beating Brock over a day or two of CPU. The full game is not realistic."],
    ["<b>B. Battles only (Showdown)</b>",
     "poke-env Python library against a local Pokemon Showdown server. Structured state (HP, types, moves, boosts); "
     "reward is win/loss.",
     "Very doable; fast on CPU. Most type-matchup and switching reasoning per training hour. <b>Needs no ROM.</b>"],
    ["<b>C. LLM agent plays Red</b>",
     "A language model reads screen + state and chooses actions, like \"Claude plays Pokemon\".",
     "Easy to run but pays API cost per step, and it is prompting rather than RL."],
], [1.45 * inch, 2.9 * inch, W - 4.35 * inch]))
s.append(p("<b>Recommendation:</b> B first, then A. B isolates the reasoning with a clean reward and needs no ROM. "
           "A is the long-horizon exploration problem and reuses B's battle lessons.", BODY))

s.append(p("The \"file that knows where everything is\" question", H2))
s.append(p("The instinct to avoid it is right, but two different things share that description. One is fine and standard; "
           "the other turns the agent into a script.", BODY))
s.append(table([
    ["Level", "What the agent gets", "Verdict"],
    ["0", "Pixels only; reward only for badges", "Fair but hopeless: reward too sparse to ever find"],
    ["1", "<b>RAM addresses used for the reward only</b>: map ID, x/y, party levels, badge bits, event flags. "
          "The agent never sees them.",
     "<font color='#1a7f37'><b>Use.</b></font> This defines the objective; it does not tell the agent how to reach it. "
     "The address table comes from the community pokered disassembly."],
    ["2", "RAM values in the observation: party HP, levels, current map, items",
     "<font color='#1a7f37'><b>Use.</b></font> A human player sees all of this on screen or in menus."],
    ["3", "A visited-tiles map the agent builds itself during the episode",
     "<font color='#1a7f37'><b>Use.</b></font> It is the agent's own memory, not outside knowledge."],
    ["4", "<b>An imported world map, route, or walkthrough</b> telling it where to go next",
     "<font color='#c0392b'><b>Do not use.</b></font> It follows an answer key, learns nothing that transfers, "
     "and eval scores measure the file instead of the agent."],
], [0.5 * inch, 2.9 * inch, W - 3.4 * inch]))
s.append(p("<b>Rule of thumb:</b> game memory may tell us <i>how the agent is doing</i> (reward) and <i>what a player could "
           "see</i> (observation), but never <i>what to do next</i>. Knowing where a value lives in RAM is a lookup "
           "table for the emulator, not memorization of the game.", CALLOUT))
s.append(p("One trap sits between levels 1 and 4: reward shaping tuned to one route. For example, paying for specific "
           "coordinates in Viridian Forest in a specific order quietly encodes the walkthrough. Keep shaped rewards "
           "generic: any new tile, any level gained, any badge, any new event flag.", BODY))

s.append(p("What carries over from mario-rl", H2))
s.append(table([
    ["Reuse as-is", "Change"],
    ["Single adapter file isolating emulator quirks", "DQN becomes PPO with parallel environments (on-policy, no replay buffer)"],
    ["Config presets with a smoke test that hits every code path", "Add recurrent memory (LSTM) or the explicit visited-tiles map"],
    ["Checkpoints that resume exactly; learning-hash guard", "Truncation is the normal case; bootstrap on it correctly"],
    ["Fixed-seed eval with held-out tests and unique-trajectory count",
     "Eval on save states the agent never trained from, or unseen Showdown teams"],
    ["Detached launcher, CSV logging, pinned environment", "Parallel envs compete with Mario for the same 16 threads"],
], [W / 2, W / 2]))

s.append(p("Before starting", H2))
s += bullets([
    "<b>ROM (option A only):</b> dump your own cartridge with a USB reader, such as an Epilogue Joey Jr. or "
    "insideGadgets GBxCart RW with FlashGBX. Back up the save too. Keep *.gb out of git; we verify the checksum "
    "before building on it.",
    "<b>CPU:</b> start after the Mario run finishes, or pause Mario first.",
    "<b>Process:</b> same as Mario. A written plan to review, then build, test, smoke run, full run.",
])

doc = SimpleDocTemplate(OUT, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                        title="Mario RL: status, decisions, and Pokemon next steps", author="NickNojiri")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
