"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

def main(args):
    all_substr = []
    for feat in args.feat:
        for category in args.category:
            all_substr.append(f"{category.upper()}_{feat}")

    download_line = []
    with open(args.src, 'r') as file:
        for line in file:
            if any(substring in line for substring in all_substr):
                download_line.append(line.strip())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(download_line)} file(s) into: {outdir}")

    for line in tqdm(download_line):
        filename = line.split("/")[-1].split("?")[0]
        out_path = outdir / filename
        # -c resumes a partial file instead of restarting it — these zips run
        # tens of GB, and the signed URLs are time-limited, not retry-limited,
        # so a dropped connection shouldn't cost the whole download.
        e = subprocess.run(["wget", "-c", "-O", str(out_path), line])
        if e.returncode != 0:
            print(f"Error downloading {line}")
            exit(1)
        print("Downloaded", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="path to the txt file containing the list of videos to download")
    parser.add_argument("--outdir", type=str, default="/data2/home/mubtasim/speech-motion/embody3d_downloads",
                        help="directory to save the downloaded zip files into (created if missing)")
    parser.add_argument("--feat", nargs="+", type=str, default=['smplx', 'audio', 'text', 'videos'], choices=['smplx', 'audio', 'text', 'videos'], help="specify the features to download")
    parser.add_argument("--category", nargs="+", type=str, default=['charades', 'daylife', 'dyadic', 'hands', "locomotion", "multiperson", "scenarios"], choices=['charades', 'daylife', 'dyadic', 'hands', "locomotion", "multiperson", "scenarios"], help="specify the category to download")
    args = parser.parse_args()
    # greps through the string and downloads only features that match your arg list.
    main(args)