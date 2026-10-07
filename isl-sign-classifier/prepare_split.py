"""Turn the official INCLUDE-50 split files into data/include50.csv.

Each line of data/splits/include50_{train,val,test}.txt is a path such as
``Greetings/48. Hello/MVI_0089.MOV`` (some are ``<Category>/<Sign>/Extra/<file>``).
Label = sign folder (second path component) without its leading number, lowercased.
"""
import argparse
import csv
import re
from pathlib import Path

SPLITS = ("train", "val", "test")


def label_of(video_path):
    sign = video_path.split("/")[1]
    return re.sub(r"^\s*\d+\s*[.\-_)]*\s*", "", sign).strip().lower()


def read_split(path):
    # Lines may end in \r, and some files lack a final newline: splitlines + strip handles both.
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--out", default="data/include50.csv")
    ap.add_argument("--categories-out", default="scripts/needed_categories.txt")
    args = ap.parse_args()

    rows = []
    for split in SPLITS:
        for p in read_split(Path(args.splits_dir) / f"include50_{split}.txt"):
            rows.append((p, label_of(p), split))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "label", "split"])
        w.writerows(rows)

    cats = sorted({r[0].split("/")[0] for r in rows})
    Path(args.categories_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.categories_out).write_text("\n".join(cats) + "\n")
    labels = {r[1] for r in rows}
    print(f"{len(rows)} videos, {len(labels)} labels -> {args.out}")
    for s in SPLITS:
        print(f"  {s}: {sum(r[2] == s for r in rows)}")
    print("Category zips needed:\n  " + "\n  ".join(cats))


if __name__ == "__main__":
    main()
