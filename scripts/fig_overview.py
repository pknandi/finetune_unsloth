"""Figure 1 (method overview) for the UpBeat paper. Schematic; no data is read.
usage: python fig_overview.py <out_dir>   -> method_overview.pdf / .png
"""
import sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 6.5
rcParams["mathtext.fontset"] = "dejavusans"

# palette
AUD = "#6A5ACD"; AUD_L = "#E9E6FA"
STU = "#138D7F"; STU_L = "#DDF3EF"
TEA = "#E07B39"; TEA_L = "#FCE8D8"
FUT = "#B8B8B8"; INK = "#2F2F2F"; MUTE = "#7A7A7A"
PANEL = "#F7F7F9"
LVL = plt.cm.Blues(np.linspace(0.85, 0.30, 8))   # coarse (dark) -> fine (light)

W, H = 140, 62
fig = plt.figure(figsize=(7.0, 7.0 * H / W))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=0.8, r=1.2, z=1, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, ls=ls)
    ax.add_patch(p); return p


def arrow(x0, y0, x1, y1, c=INK, lw=0.9, z=5, style="-|>", rad=0.0, ms=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=c, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def txt(x, y, s, size=6.5, c=INK, ha="center", va="center", w="normal", z=6, style="normal"):
    ax.text(x, y, s, fontsize=size, color=c, ha=ha, va=va, fontweight=w, zorder=z, style=style)


rng = np.random.default_rng(3)
def speech(n):  # syllable-like envelope x carrier
    t = np.linspace(0, 1, n)
    env = np.clip(np.sin(np.pi * t * 7) ** 2 * (0.5 + 0.5 * np.sin(t * 13 + 1)) + 0.08, 0, 1)
    return env * np.sin(t * 260) * (0.6 + 0.4 * rng.random(n))
WAVE = speech(420)


def waveform(x0, x1, yc, amp, now=None, past_c=AUD, fut_c=FUT, lw=0.5, fut_ls="-"):
    xs = np.linspace(x0, x1, len(WAVE))
    if now is None:
        ax.plot(xs, yc + amp * WAVE, color=past_c, lw=lw, zorder=4); return
    m = xs <= now
    ax.plot(xs[m], yc + amp * WAVE[m], color=past_c, lw=lw, zorder=4)
    ax.plot(xs[~m], yc + amp * WAVE[~m], color=fut_c, lw=lw, zorder=4, ls=fut_ls)


def tokens(x, y, ncol, cw=1.55, ch=1.05, gap=0.22, hl_last=True, alpha_last=1.0):
    for c in range(ncol):
        for q in range(8):
            a = alpha_last if (c == ncol - 1) else 1.0
            ax.add_patch(Rectangle((x + c * (cw + gap), y + (7 - q) * (ch + gap)), cw, ch, fc=LVL[q], ec="white",
                                   lw=0.3, alpha=a, zorder=4))
    if hl_last:
        c = ncol - 1
        ax.add_patch(Rectangle((x + c * (cw + gap) - 0.25, y - 0.25), cw + 0.5, 8 * (ch + gap) - gap + 0.5, fc="none",
                               ec=STU, lw=0.9, zorder=5))
    return x + ncol * (cw + gap) - gap, y + 8 * (ch + gap) - gap


def figure_pose(cx, base, s, arms, c, alpha=1.0, lw=1.3):
    """Stick figure: arms = (l_up, l_lo, r_up, r_lo) angles in degrees from straight down."""
    hip = np.array([cx, base + 4.2 * s]); neck = hip + [0, 3.6 * s]; head = neck + [0, 1.25 * s]
    kw = dict(color=c, lw=lw, alpha=alpha, solid_capstyle="round", zorder=6)
    for dx in (-0.8, 0.8):
        ax.plot([hip[0], hip[0] + dx * s], [hip[1], base], **kw)
    ax.plot([hip[0], neck[0]], [hip[1], neck[1]], **kw)
    sh = neck - [0, 0.35 * s]
    for side, (u, l) in ((-1, arms[:2]), (1, arms[2:])):
        a1 = np.deg2rad(u); a2 = np.deg2rad(l)
        el = sh + 1.9 * s * np.array([side * np.sin(a1), -np.cos(a1)])
        wr = el + 1.7 * s * np.array([side * np.sin(a2), -np.cos(a2)])
        ax.plot([sh[0], el[0], wr[0]], [sh[1], el[1], wr[1]], **kw)
    ax.add_patch(Circle(head, 0.85 * s, fc="white", ec=c, lw=lw, alpha=alpha, zorder=6))


def layers(x, y, c, fill_top, w=4.6, h=1.3, n=3, dx=0.55, dy=0.95):
    """Stack of n slanted slabs = a neural network (transformer layers)."""
    for i in range(n):
        top = i == n - 1
        box(x + i * dx, y + i * dy, w, h, c if top and fill_top else "white", c, lw=0.7, r=0.35, z=5)


def causal_tri(x, y, sz, c):
    """Lower-triangular attention mask: frame t attends to frames <= t."""
    ax.add_patch(Rectangle((x, y), sz, sz, fc="white", ec=c, lw=0.6, zorder=5))
    ax.add_patch(Polygon([[x, y], [x, y + sz], [x + sz, y]], closed=True, fc=c, ec="none", zorder=6))


def panel(x, y, w, h, title):
    box(x, y, w, h, PANEL, "#D9D9DF", lw=0.6, r=1.6, z=0)
    txt(x + 1.6, y + h - 2.0, title, size=7.2, ha="left", w="bold")


# ---------------------------------------------------------------- (a) streaming inference
panel(1, 33, 138, 28, "(a) Streaming inference with zero audio look-ahead")

# live speech
ax_now = 19.5
waveform(3.5, 24.5, 47.5, 3.6, now=ax_now, fut_ls=(0, (1.2, 1.2)))
ax.plot([ax_now, ax_now], [42.3, 53.0], color=INK, lw=0.7, zorder=5)
txt(ax_now - 0.6, 54.0, "now ($t$)", size=6, ha="right")
txt(11.0, 40.6, "speech heard so far", size=5.6, c=AUD)
txt(20.2, 40.4, "future:\nunused", size=5.0, c=MUTE, style="italic", ha="left")

# streamed HuBERT: staircase, one pass per slot
arrow(25.5, 47.5, 28.8, 47.5)
sx, sy, n = 29.5, 41.6, 6
for j in range(n):
    for k in range(j + 1):
        ax.add_patch(Rectangle((sx + k * 1.55, sy + (n - 1 - j) * 1.55), 1.35, 1.35,
                               fc=AUD if k == j else AUD_L, ec="white", lw=0.3, zorder=4))
txt(sx + 3.9, 52.6, "streamed HuBERT", size=5.9, w="bold")
txt(sx + 4.6, 39.6, "one pass per slot,\naudio up to its end", size=5.5, c=MUTE)

# causal AR transformer
arrow(39.9, 47.5, 42.4, 47.5)
box(42.6, 42.0, 23.8, 11.5, STU_L, STU, lw=0.9)
txt(54.5, 51.9, "causal AR transformer", size=6.0, w="bold", c=STU)
txt(54.5, 49.0, "audio encoder, motion decoder,\ndepth head over RVQ levels", size=4.9, c=INK)
# icons: transformer layers + causal attention mask
layers(44.3, 42.7, STU, True)
causal_tri(51.2, 42.9, 2.6, STU)
txt(54.3, 44.2, "causal attention:\nsees only the past", size=4.6, c=STU, ha="left")

# adaptive guidance (below transformer)
box(42.5, 34.6, 24.0, 6.0, "#FFF7DA", "#D8A800", lw=0.8)
txt(54.5, 39.1, "uncertainty-adaptive guidance", size=4.9, w="bold", c="#8A6A00")
txt(54.5, 36.6, r"$g_t = g_{\min} + (g_{\max}-g_{\min})\,H_t$", size=5.8, c=INK)
arrow(54.5, 40.6, 54.5, 41.95, c="#B58E00", lw=0.8)

# RVQ tokens: past frames + the next frame's 8 residual levels, predicted coarse to fine
arrow(66.6, 47.5, 68.6, 47.5)
txt(84.6, 54.7, "RVQ motion tokens", size=6.2, w="bold")
cw, ch, gp = 1.25, 1.05, 0.22
for c in range(4):                                   # past frames (already generated)
    for q in range(8):
        ax.add_patch(Rectangle((69.0 + c * (cw + gp), 42.4 + (7 - q) * (ch + gp)), cw, ch, fc=LVL[q], ec="white",
                               lw=0.3, alpha=0.55, zorder=4))
txt(71.6, 53.3, "past frames", size=4.4, c=MUTE)
nx = 75.8; ncw = 2.1                                 # next frame, enlarged
for q in range(8):
    ax.add_patch(Rectangle((nx, 42.4 + (7 - q) * (ch + gp)), ncw, ch, fc=LVL[q], ec="white", lw=0.3, zorder=4))
ax.add_patch(Rectangle((nx - 0.25, 42.15), ncw + 0.5, 8 * (ch + gp) - gp + 0.5, fc="none", ec=STU, lw=0.9, zorder=5))
# depth head: level by level, coarse to fine; each level codes what the earlier levels missed
arrow(78.9, 51.8, 78.9, 43.0, c=STU, lw=0.7, ms=5)
txt(79.5, 52.1, "$q{=}1$ coarse", size=4.6, c=LVL[0], ha="left")
txt(79.5, 49.6, "depth head,\nlevel by level", size=4.4, c=STU, ha="left")
txt(79.5, 42.9, "$q{=}8$ fine", size=4.6, c=LVL[4], ha="left")
txt(77.4, 40.9, r"$z_t = e_1(k_{t,1}) + \dots + e_8(k_{t,8})$", size=5.2, c=INK)
txt(79.8, 39.1, "8 residual levels $\\times$ 1024 codes, 15 fps", size=4.4, c=MUTE)
# autoregressive loop back into the transformer
arrow(nx + 0.3, 53.0, 61.0, 53.8, c=STU, lw=0.7, rad=0.45, style="-|>")
txt(69.5, 59.0, "next frame fed back", size=5.2, c=STU)

# energy-guided chunk selection (decoding)
box(68.0, 33.7, 28.6, 4.3, "#EFEAFB", "#7A68C9", lw=0.8)
txt(82.3, 36.6, "distilled causal energy", size=5.2, w="bold", c="#5B4BB0")
txt(82.3, 34.7, "keeps the best of $K{=}4$ sampled 1 s chunks", size=4.4, c=INK)

# causal decoder
arrow(80.2, 45.8, 89.4, 45.8)
ax.add_patch(Polygon([[89.6, 51.0], [89.6, 42.8], [97.2, 44.0], [97.2, 49.8]], closed=True, fc=STU_L, ec=STU, lw=0.9,
                     zorder=4, joinstyle="round"))
txt(93.4, 47.6, "causal\ndecoder", size=5.3, w="bold", c=STU)
txt(93.4, 41.2, "0 / 67 / 133 ms\noutput latency", size=4.8, c=MUTE)

# stream of poses
arrow(97.6, 46.9, 100.0, 46.9)
poses = [(12, 10, 12, 10), (35, 70, 18, 25), (55, 110, 30, 70), (20, 40, 60, 115), (15, 25, 25, 45)]
for i, a in enumerate(poses):
    figure_pose(103.4 + i * 7.0, 41.6, 1.05, a, STU, alpha=0.35 + 0.65 * (i + 1) / len(poses))
ax.plot([100.4, 137.0], [41.3, 41.3], color=MUTE, lw=0.5, zorder=3)
txt(118.7, 39.4, "continuous stream: no real-motion seed, no resets", size=5.3, c=MUTE)
txt(118.7, 55.4, "SMPL-X body and hands, 30 fps", size=6.2, w="bold")

# ---------------------------------------------------------------- (b) distillation
panel(1, 1, 86, 30.5, "(b) Privileged-future distillation on deployed rollouts")

# teacher row
box(3.2, 16.6, 23.0, 10.4, TEA_L, TEA, lw=0.9)
txt(4.4, 25.4, "offline teacher", size=6.3, w="bold", c=TEA, ha="left")
txt(4.4, 23.6, "frozen", size=4.8, c=MUTE, ha="left", style="italic")
layers(4.4, 17.6, TEA, True, w=3.6, h=1.1, dx=0.45, dy=0.85)
waveform(10.2, 24.8, 20.3, 1.9, now=19.2, past_c=TEA, fut_c=TEA)
ax.add_patch(Rectangle((19.2, 18.0), 5.6, 4.6, fc=TEA, alpha=0.12, lw=0, zorder=3))
ax.plot([19.2, 19.2], [18.0, 22.8], color=INK, lw=0.6, zorder=5)
txt(25.0, 23.6, "hears the future", size=5.0, c=TEA, ha="right")

# student row
box(3.2, 2.6, 23.0, 10.4, STU_L, STU, lw=0.9)
txt(4.4, 11.4, "streaming student", size=6.3, w="bold", c=STU, ha="left")
txt(4.4, 9.4, "trained", size=4.8, c=MUTE, ha="left", style="italic")
layers(4.4, 3.6, STU, True, w=3.6, h=1.1, dx=0.45, dy=0.85)
waveform(10.2, 24.8, 5.6, 1.9, now=19.2, past_c=STU, fut_c=FUT, fut_ls=(0, (1.2, 1.2)))
ax.plot([19.2, 19.2], [3.3, 8.0], color=INK, lw=0.6, zorder=5)
txt(25.0, 9.4, "audio up to $t$", size=5.0, c=MUTE, ha="right")

# rollout tokens
arrow(26.6, 7.6, 31.6, 11.3, c=STU, lw=0.9)
txt(27.5, 4.6, "sample as deployed\n($T{=}1.3$, guidance 3.5)", size=5.2, c=STU, ha="left")
rx1, ry1 = tokens(32.0, 9.6, 5, cw=1.45, ch=0.92, hl_last=False)
txt(35.9, 21.6, "student's own\nrollout", size=5.4, c=INK)
# teacher scores the rollout
arrow(26.6, 22.0, 32.0, 20.0, c=TEA, lw=0.9)
txt(29.6, 26.6, "scores every token\n(teacher forcing)", size=5.2, c=TEA, ha="left")

# distributions
arrow(rx1 + 0.6, 14.5, 46.4, 14.5)
bx, by = 47.2, 6.6
pt = np.array([0.08, 0.34, 0.27, 0.12, 0.09, 0.06, 0.04])
ps = np.array([0.15, 0.22, 0.20, 0.16, 0.12, 0.09, 0.06])
for i in range(len(pt)):
    ax.add_patch(Rectangle((bx + i * 2.0, by), 0.85, pt[i] * 34, fc=TEA, alpha=0.85, lw=0, zorder=4))
    ax.add_patch(Rectangle((bx + i * 2.0 + 0.85, by), 0.85, ps[i] * 34, fc=STU, alpha=0.85, lw=0, zorder=4))
ax.plot([bx - 0.4, bx + 14.2], [by, by], color=INK, lw=0.5, zorder=5)
txt(bx + 6.9, by - 1.5, "codes of one frame and level", size=5.0, c=MUTE)
txt(bx + 6.9, 21.3, "guided teacher $\\pi_T$", size=5.6, c=TEA)
txt(bx + 6.9, 19.6, "vs. student $p_S$", size=5.6, c=STU)

# loss
box(63.8, 8.6, 22.4, 16.0, "white", "#C9C9D1", lw=0.6)
txt(74.8, 21.8, r"$\sum_{t,q}\; w_q\,\mathrm{KL}\!\left(\pi_T^{(t,q)}\,\|\,p_S^{(t,q)}\right)$", size=6.6)
txt(74.8, 18.6, "+ likelihood of real motion", size=5.2, c=MUTE)
txt(74.8, 16.6, "exact in token space, no critic", size=5.0, c=MUTE)
wq = [2, 2, 1, 1, 1, 1, 0.5, 0.5]
for q, w in enumerate(wq):
    ax.add_patch(Rectangle((67.3 + q * 1.95, 10.2), 1.4, w * 2.1, fc=LVL[q], ec="none", zorder=4))
txt(74.8, 7.2, "level weights $w_q$: coarse > fine", size=5.2, c=MUTE)
arrow(61.9, 14.5, 63.6, 14.5)

# ---------------------------------------------------------------- (c) measurement
panel(89, 1, 50, 30.5, "(c) Measuring look-ahead and timing")

# audit (top half)
txt(91.0, 26.6, "look-ahead audit", size=6.2, w="bold", ha="left")
xs = np.linspace(91.5, 112.5, len(WAVE)); tau = 104.0
m = xs <= tau
ax.plot(xs[m], 21.7 + 1.7 * WAVE[m], color=AUD, lw=0.5, zorder=4)
ax.plot(xs[~m], 21.7 + 1.7 * rng.normal(0, 0.45, (~m).sum()), color="#C0392B", lw=0.4, zorder=4)
ax.plot([tau, tau], [19.3, 24.2], color=INK, lw=0.6, zorder=5)
txt(tau + 0.6, 24.1, r"$\tau$", size=6, ha="left")
txt(109.0, 18.3, "noise after $\\tau$", size=5.0, c="#C0392B")
txt(97.0, 18.3, "output before $\\tau$\nmust not move", size=5.0, c=INK)
# results bars
rows = [("full-window HuBERT", 6.93, "#C0392B"), ("Super Star tokens", 6.96, "#C0392B"),
        ("decoder + smoothing", 0.467, "#C0392B"), ("UpBeat, audio path", 0.0, STU)]
for i, (lab, v, c) in enumerate(rows):
    yy = 26.0 - i * 2.3
    txt(114.2, yy, lab, size=5.0, ha="left", c=INK)
    txt(137.6, yy, ("%.2f s" % v) if v > 0 else r"$\leq$0 s", size=5.0, ha="right", c=c, w="bold")

# lead/lag (bottom half)
txt(91.0, 14.4, "speech-motion timing (lead / lag)", size=6.2, w="bold", ha="left")
txt(91.0, 12.4, "correlation of loudness and motion speed vs. time shift", size=4.8, c=MUTE, ha="left")
lx0, lx1, ly0 = 93.0, 136.0, 3.2
lag = np.linspace(-400, 300, 300)
def to_x(v): return lx0 + (v + 400) / 700 * (lx1 - lx0)
curves = [(-167, "#3A3A3A", "people  $-$167 ms", 1.0), (33, "#9E9E9E", "streaming base  +33", 0.85),
          (0, STU, "UpBeat  0", 1.0)]
for pk, c, lab, a in curves:
    y = np.exp(-0.5 * ((lag - pk) / 110) ** 2)
    ax.plot(to_x(lag), ly0 + 6.0 * y, color=c, lw=1.0, alpha=a, zorder=5)
    ax.add_patch(Circle((to_x(pk), ly0 + 6.0), 0.45, fc=c, ec="white", lw=0.4, zorder=7))
ax.plot([lx0, lx1], [ly0, ly0], color=INK, lw=0.5, zorder=4)
ax.plot([to_x(0)] * 2, [ly0 - 0.4, ly0 + 7.4], color=INK, lw=0.5, ls=(0, (2, 1.5)), zorder=4)
txt(to_x(0), ly0 - 1.3, "0", size=5.0, c=INK)
txt(lx0 - 1.0, ly0 - 1.3, "motion leads speech", size=5.0, c=MUTE, ha="left")
txt(lx1 + 1.0, ly0 - 1.3, "motion lags speech", size=5.0, c=MUTE, ha="right")
txt(to_x(-167) - 0.5, ly0 + 7.3, "people $-$167 ms", size=5.0, c="#3A3A3A", ha="right")
txt(to_x(0) - 0.6, ly0 + 7.3, "UpBeat 0 ms", size=5.0, c=STU, ha="right")
txt(to_x(150), ly0 + 5.4, "base +33", size=5.0, c=MUTE, ha="left")

out = sys.argv[1] if len(sys.argv) > 1 else "."
fig.savefig(f"{out}/method_overview.pdf")
fig.savefig(f"{out}/method_overview.png", dpi=300)
print("saved", out)
