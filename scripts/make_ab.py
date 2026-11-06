"""User-study A/B videos from the rendered clips (scripts/render_clips.sh, scripts/render_embody_clips.sh): for each
picked clip and each method pair, two clips side by side labelled only "A" / "B", left/right random (seed 0), the
clip's audio. Writes <out>/<dataset>_<clip>_<pair>.mp4, <out>/key.csv (which method is A / B; keep away from raters)
and <out>/groups.csv: 3 rater groups in a Latin square, each group sees every clip once under one pair, so each pair
gets 4 clips per dataset per group. usage: python scripts/make_ab.py [out_dir]
"""
import os, sys, csv, random, subprocess
FF = "/data/home/mubtasim/conda-envs/motion/bin/ffmpeg"
U = "outputs/userstudy-nov04"
out = sys.argv[1] if len(sys.argv) > 1 else f"{U}/ab"
os.makedirs(out, exist_ok=True)
SETS = {
    "beat2": ([f"2_scott_0_{k}_{k}" for k in (1, 2, 3, 4, 5, 6, 8, 65, 81, 87, 95, 103)],
              lambda c, m: f"{U}/clips/{c}_{m}.mp4", [("ours_f2", "emage"), ("ours_f2", "lom"), ("ours_f2", "gt")]),
    "embody": ([f"val{k}" for k in (82, 150, 209, 227, 256, 265, 340, 366, 394, 444, 497, 501)],
               lambda c, m: f"{U}/embody/clips/{c}_{m}.mp4", [("ours_f2", "emage"), ("ours_f2", "gt"), ("ours_ol", "emage")]),
}
rng = random.Random(0)
key, groups = [], []
for ds, (clips, path, pairs) in SETS.items():
    for ci, c in enumerate(clips):
        for pi, (m1, m2) in enumerate(pairs):
            a, b = (m1, m2) if rng.random() < 0.5 else (m2, m1)
            name = f"{ds}_{c}_{m1}-vs-{m2}.mp4"
            lab = lambda k, t: f"[{k}:v]pad=iw+16:ih:8:0:white,drawtext=text='{t}':x=(w-tw)/2:y=12:fontsize=36:fontcolor=black[v{k}]"
            subprocess.run([FF, "-y", "-loglevel", "error", "-i", path(c, a), "-i", path(c, b), "-filter_complex",
                            f"{lab(0, 'A')};{lab(1, 'B')};[v0][v1]hstack=2[v]", "-map", "[v]", "-map", "0:a",
                            "-c:v", "libx264", "-crf", "23", "-c:a", "copy", f"{out}/{name}"], check=True)
            key.append({"file": name, "dataset": ds, "clip": c, "pair": f"{m1}-vs-{m2}", "A": a, "B": b})
            for g in range(3):
                if (ci + g) % 3 == pi:
                    groups.append({"group": g + 1, "file": name})
        print(ds, c, flush=True)
for fn, rows in (("key.csv", key), ("groups.csv", groups)):
    with open(f"{out}/{fn}", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(key)} A/B videos, {len(groups)} group entries -> {out}")
