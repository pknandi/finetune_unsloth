"""Paper tables from the raw eval logs (outputs/fix-oct21), so every number in the paper is regenerated from source.
Writes docs/PAPER_RESULTS.md (Markdown tables, each row with its source log) and docs/tables/*.tex (LaTeX rows).
Embody rows: ALL-CLIP speech match (clip_top5, eval matcher), FD, jerk ratio to real (1 = real), per decoding mode,
mean (and std) over seeded draws; pooled rows concatenate the draws of several logs (e.g. eval groups 1-4).
BEAT2 rows: test FGD / BC over the seeded draws (f15s*), unseeded draw separately.
Constants (published or measured once and logged) are marked as such with their EXPERIMENT_LOG date.

usage: python scripts/collect_results.py
"""
import os, re, glob, statistics as st
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
F = "outputs/fix-oct21"


def e60(logs, prefix="sample1.3p1f15g3.5"):
    """pooled draws of one decoding mode across logs -> dict of lists"""
    out = {"top5": [], "fd": [], "jerk": [], "speak": [], "listen": []}
    for lg in logs:
        p = f"{F}/{lg}.log"
        if not os.path.exists(p):
            return None
        for l in open(p):
            m = re.match(rf"^(HELDOUT|DIST|MATCH|MATCHSUB) mode=({re.escape(prefix)}(?:s\d+)?) (.*)", l)
            if not m:
                continue
            kind, rest = m.group(1), m.group(3)
            if kind == "MATCH":
                out["top5"].append(float(re.search(r"clip_top5=([0-9.]+)", rest).group(1)))
            elif kind == "DIST":
                out["fd"].append(float(re.search(r"fd=([0-9.]+)", rest).group(1)))
            elif kind == "HELDOUT":
                out["jerk"].append(float(re.search(r"jerk=([0-9.]+)", rest).group(1)))
            elif kind == "MATCHSUB":
                sub = re.search(r"sub=(\w+)", rest).group(1)
                out[sub].append(float(re.search(r"clip_top5=([0-9.]+)", rest).group(1)))
    return out if out["top5"] else None


def beat2(tag):
    p = f"{F}/ol_stream_beat2_ft_{tag}.log"
    if not os.path.exists(p):
        return None
    seeded, unseeded, bc = [], [], []
    for l in open(p):
        m = re.match(r"^test f(\d+)s(\d+) .*fgd=([0-9.]+) bc=([0-9.]+)", l)
        if m:
            (seeded if m.group(1) != "0" else unseeded).append(float(m.group(3)))
            if m.group(1) != "0":
                bc.append(float(m.group(4)))
    return {"fgd": seeded, "unseeded": unseeded, "bc": bc} if seeded else None


ms = lambda v, d=3: "-" if not v else (f"{st.mean(v):.{d}f}" + (f" ± {st.stdev(v):.{d}f}" if len(v) > 1 else ""))
mean = lambda v, d=3: "-" if not v else f"{st.mean(v):.{d}f}"

# ---------------------------------------------------------------- tables
EMB_MAIN = [  # label, logs, mode prefix, note
    ("EMAGE (offline, retrained on Embody-3D)", [f"e60_emage_grp{g}" for g in (1, 2, 3, 4)], "emage", "eval groups 1-4, 1 sample"),
    ("Ours offline teacher F2 (old decoder)", [f"e60_dt_F2_grp{g}" for g in (1, 2, 3, 4)], "sample1.3p1f15g3.5", "eval groups 1-4"),
    ("Online base (causal decoder)", [f"e60_dt_OLs24_causal_grp{g}" for g in (1, 2, 3, 4)], "sample1.3p1f15g3.5", "eval groups 1-4"),
    ("Online + ours, Db (causal decoder)", [f"e60_dt_Db_nov06_causal_grp{g}" for g in (1, 2, 3, 4)], "sample1.3p1f15g3.5", "eval groups 1-4"),
    ("Online + ours, Db + adaptive guidance 1.5-5", ["e60_dt_Db_adapt"] + [f"e60_dt_Db_adapt_grp{g}" for g in (2, 3, 4)], "sample1.3p1f15a1.5-5", "eval groups 1-4"),
]
EMB_ABL = [  # group 0 (the group recipes were chosen on), causal decoder unless noted
    ("Base, old symmetric decoder (467 ms output look-ahead)", ["e60_dt_OLs24_stream", "e60_dt_OLs24_more_stream"], "sample1.3p1f15g3.5"),
    ("Base, causal decoder", ["e60_dt_OLs24_causal_s25_stream"], "sample1.3p1f15g3.5"),
    ("+ forward-KL on-policy distillation (d4)", ["e60_dt_OLd4_causal_s25_stream"], "sample1.3p1f15g3.5"),
    ("+ deployment-matched rollouts (Ca)", ["e60_dt_Ca_nov06_causal_stream"], "sample1.3p1f15g3.5"),
    ("+ guided teacher target (Cb)", ["e60_dt_Cb_nov06_causal_stream"], "sample1.3p1f15g3.5"),
    ("+ level-aware weights (Db)", ["e60_dt_Db_nov06_causal_stream"], "sample1.3p1f15g3.5"),
    ("Db, seed 1", ["e60_dt_Db_s1_nov06_causal_stream"], "sample1.3p1f15g3.5"),
    ("Db, one forward pass (guidance 1)", ["e60_dt_Db_nov06_causal_stream"], "sample1.3p1f15g1"),
    ("Cb + speech-salient weights (Da, dropped)", ["e60_dt_Da_nov06_causal_stream"], "sample1.3p1f15g3.5"),
    ("Db + anticipatory encoder loss (E1, dropped)", ["e60_dt_E1_nov06_causal_stream"], "sample1.3p1f15g3.5"),
    ("Db, 2-token decoder (133 ms)", ["e60_dt_Db_nov06_cf2_stream"], "sample1.3p1f15g3.5"),
    ("Db, jitter-free decoder (0 ms)", ["e60_dt_Db_nov06_jf0_stream"], "sample1.3p1f15g3.5"),
    ("Db, jitter-free decoder (133 ms)", ["e60_dt_Db_nov06_jf2_stream"], "sample1.3p1f15g3.5"),
    ("Offline teacher F2 (old decoder, hears the future)", ["e60_dt_F2_cfg", "e60_dt_F2_more"], "sample1.3p1f15g3.5"),
]
EMB_DECODE = [  # eval group 1 (fresh), Db
    ("Db, fixed guidance 3.5", ["e60_dt_Db_nov06_causal_grp1"], "sample1.3p1f15g3.5"),
    ("Db + energy-guided decoding, causal (K=4)", ["e60_dt_Db_e4c"], "sample1.3p1f15g3.5e4c"),
    ("Db + energy-guided decoding, oracle +160 ms (analysis)", ["e60_dt_Db_e4o8"], "sample1.3p1f15g3.5e4o8"),
    ("Db + adaptive guidance 1.5-5", ["e60_dt_Db_adapt"], "sample1.3p1f15a1.5-5"),
    ("Db + adaptive guidance 1-7", ["e60_dt_Db_adapt"], "sample1.3p1f15a1-7"),
]
EMB_LAT = [  # one model any latency vs dedicated models (old decoder, base model, group 0)
    ("0 ms: dedicated", ["e60_dt_OLs24_stream"], "sample1.3p1f15g3.5"), ("0 ms: one model", ["e60_dt_ANY_s0_stream"], "sample1.3p1f15g3.5"),
    ("120 ms: dedicated", ["e60_dt_OLs24_la3"], "sample1.3p1f15g3.5"), ("120 ms: one model", ["e60_dt_ANY_s3_stream"], "sample1.3p1f15g3.5"),
    ("280 ms: dedicated", ["e60_dt_OLs24_la7"], "sample1.3p1f15g3.5"), ("280 ms: one model", ["e60_dt_ANY_s7_stream"], "sample1.3p1f15g3.5"),
    ("520 ms: dedicated", ["e60_dt_OLs24_la13"], "sample1.3p1f15g3.5"), ("520 ms: one model", ["e60_dt_ANY_s13_stream"], "sample1.3p1f15g3.5"),
]
B2 = [  # label, tag
    ("Online base (old decoder)", "ols24"), ("Online base, earlier run (old decoder)", None),
    ("+ forward-KL distillation, seed 0 (old decoder, 2-sided smoothing)", "b2dfwd"),
    ("+ forward-KL distillation, seed 1", "b2dfwds1"), ("+ forward-KL distillation, seed 2", "b2dfwds2"),
    ("seed 0, old decoder, 1-sided smoothing", "b2dfwd_old_s1side"), ("seed 0, old decoder, no smoothing", "b2dfwd_old_s0"),
    ("seed 0, causal decoder 0 ms", "b2dfwd_causal"), ("seed 0, causal decoder 67 ms", "b2dfwd_causal_f1"),
    ("seed 0, causal decoder 133 ms", "b2dfwd_causal_f2"), ("seed 1, causal decoder 133 ms", "b2dfwds1_cf2"),
    ("seed 2, causal decoder 133 ms", "b2dfwds2_cf2"), ("C/D (Db recipe), old decoder", "b2Dblast"),
    ("C/D (Db recipe), causal 133 ms", "b2Dblast_cf2"), ("Undistilled online, causal 133 ms", "b2olscf2"),
]
B2_CONST = [("EMAGE (offline; scored with our tools)", "0.615", "0.591", "11-01 log"),
            ("LoM (offline, released; our tools)", "0.517", "0.671", "11-02 log"),
            ("Ours offline F2 (5 draws)", "0.575", "-", "11-04 log"),
            ("Honest online base (ols, 5 draws)", "0.655 ± 0.012", "0.766", "11-04 log, ol_stream_beat2_ft.log")]


def table_emb(rows, title, cols=("speech", "speak", "listen", "FD", "jerk")):
    out = [f"### {title}", "", "| Model | Speech match ↑ | Speaking | Listening | FD ↓ | Jerk ratio (1 = real) | Draws | Source |",
           "|---|---|---|---|---|---|---|---|"]
    tex = []
    for r in rows:
        label, logs, pre = r[0], r[1], r[2]
        d = e60(logs, pre)
        if d is None:
            out.append(f"| {label} | pending | | | | | | {', '.join(logs)} |"); continue
        out.append(f"| {label} | {ms(d['top5'])} | {mean(d['speak'])} | {mean(d['listen'])} | {ms(d['fd'], 2)} | "
                   f"{mean(d['jerk'], 2)} | {len(d['top5'])} | {', '.join(logs)} [{pre}] |")
        tex.append(f"{label} & {mean(d['top5'])} & {mean(d['fd'], 2)} & {mean(d['jerk'], 2)} \\\\")
    return out + [""], tex


def table_b2():
    out = ["### BEAT2 (speaker 2 test, 5 seeded draws; val-picked guidance)", "",
           "| Model | FGD ↓ | BC ↑ | Unseeded FGD | Source |", "|---|---|---|---|---|"]
    tex = []
    for label, tag in B2:
        if tag is None:
            continue
        d = beat2(tag)
        if d is None:
            out.append(f"| {label} | pending | | | ol_stream_beat2_ft_{tag}.log |"); continue
        out.append(f"| {label} | {ms(d['fgd'])} | {mean(d['bc'])} | {mean(d['unseeded'])} | ol_stream_beat2_ft_{tag}.log |")
        tex.append(f"{label} & {mean(d['fgd'])} & {mean(d['bc'])} \\\\")
    for label, fgd, bc, src in B2_CONST:
        out.append(f"| {label} | {fgd} | {bc} | | constant, {src} |")
    return out + [""], tex


def cont():
    out = ["### Continuous 60 s streaming (20 val tracks, no real seed)", "",
           "| Model | Mode | FD per 10 s segment | Seam jump (1 = ordinary frame) | Source |", "|---|---|---|---|---|"]
    for name in ("base", "Db"):
        p = f"{F}/cont_{name}.log"
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        for mode in ("stream", "reset"):
            fds = re.findall(rf"CONT mode={mode} seg=\S+ fd=([0-9.]+)", txt)
            sj = re.search(rf"CONT mode={mode} boundary_jump=([0-9.]+)", txt)
            out.append(f"| {name} | {mode} | {' / '.join(fds)} | {sj.group(1) if sj else '-'} | cont_{name}.log |")
    return out + [""]


md = ["# Paper results", "", "Generated by `scripts/collect_results.py` from the raw logs in `outputs/fix-oct21/`. Do not edit by hand;",
      "re-run the script. Speech match = ALL-CLIP clip_top5 (eval matcher, seed 0); FD = Fréchet distance of pose PCA +",
      "speed features to the same windows' real motion; jerk ratio = generated / real jerk.", ""]
os.makedirs("docs/tables", exist_ok=True)
for rows, title, name in ((EMB_MAIN, "Embody-3D main (fresh eval groups 1-4, 60 windows each, causal decoder)", "embody_main"),
                          (EMB_ABL, "Embody-3D ablation (eval group 0, used to choose recipes; causal decoder unless noted)", "embody_ablation"),
                          (EMB_DECODE, "Decoding: energy-guided and adaptive guidance (fresh eval group 1)", "embody_decoding"),
                          (EMB_LAT, "One model, any latency vs dedicated models (old decoder, group 0)", "embody_latency")):
    t, tex = table_emb(rows, title); md += t
    open(f"docs/tables/{name}.tex", "w").write("\n".join(tex) + "\n")
t, tex = table_b2(); md += t; open("docs/tables/beat2.tex", "w").write("\n".join(tex) + "\n")
md += cont()
md += ["### Signed lead/lag of motion vs own speech (scripts/lead_lag.py; negative = motion leads)", "",
       "Logged values (EXPERIMENT_LOG 2026-11-06 12:20): real -167 ms [-300,-67] (531 val windows); offline F2 -133 ms;",
       "online base with old symmetric decoder -33 ms; online base causal +67 ms [+33,+100]; Db causal +33 ms [0,+33].",
       "Re-run: `python scripts/lead_lag.py outputs/fix-oct21/<eval dir> <mode prefix>`.", ""]
open("docs/PAPER_RESULTS.md", "w").write("\n".join(md) + "\n")
print("wrote docs/PAPER_RESULTS.md and docs/tables/*.tex")
