"""Paper figure (R12): audio look-ahead measured by the audit (scripts/leak_probe.py): output changes when the audio
after a cut is replaced by noise. Values (seconds, max over windows and cuts; <= 0 = zero look-ahead): full-window
HuBERT for the offline teacher and for the online model 6.93; UpBeat with streamed features -0.067; the fixed
look-ahead variants (feature shifts 0.12 / 0.28 / 0.52 s) measure 0.067 / 0.200 / 0.467 (quantised to motion slots).
Horizontal bars on a symmetric-log axis so the 6.9 s and the sub-second values are both readable.

usage: python scripts/fig_leak.py [out=outputs/figures-paper/leak_probe.pdf]
"""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/figures-paper/leak_probe.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
rows = [("Full-window HuBERT (offline teacher)", 6.93, "#C0392B"),
        ("Full-window HuBERT (online model)", 6.93, "#C0392B"),
        ("UpBeat (streamed)", -0.067, "#138D7F"),
        ("+0.12 s look-ahead variant", 0.067, "#9E9E9E"),
        ("+0.28 s look-ahead variant", 0.200, "#9E9E9E"),
        ("+0.52 s look-ahead variant", 0.467, "#9E9E9E")]
fig, ax = plt.subplots(figsize=(3.3, 1.8))
y = list(range(len(rows)))[::-1]
for yy, (name, v, c) in zip(y, rows):
    ax.barh(yy, v, color=c, height=0.62)
    lab = f"{v:.2f} s" if abs(v) >= 1 else f"{v:+.3f} s"
    ax.text(v * 1.15 + 0.012 if v > 0 else 0.015, yy, lab, va="center", ha="left", fontsize=5.6, color=c)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.8)
ax.set_xscale("symlog", linthresh=0.1); ax.set_xlim(-0.4, 25)
ax.set_xticks([-0.1, 0, 0.1, 1, 10]); ax.set_xticklabels(["-0.1", "0", "0.1", "1", "10"], fontsize=5.8)
ax.axvline(0, color="#555555", lw=0.7)
ax.set_xlabel("audio look-ahead used (s)", fontsize=6.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=220)
print("wrote", out)
