"""Videos -> per-video landmark arrays ``(frames, 61, 2)`` plus ``index.csv``.

Row layout per frame: 0-18 pose (indices 0..16, 23, 24), 19-39 LEFT hand,
40-60 RIGHT hand (person's left/right). Missing points are NaN.
"""
import argparse
import csv
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
POSE_IDX = list(range(17)) + [23, 24]
MAX_WIDTH = 640
MISSING_WRIST_COST = 1.0  # cost of putting a hand on a slot whose pose wrist is missing


def download_models(model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for url in (POSE_MODEL_URL, HAND_MODEL_URL):
        dest = model_dir / url.rsplit("/", 1)[-1]
        if not dest.exists():
            print(f"Downloading {url}", file=sys.stderr)
            tmp = dest.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
        paths.append(str(dest))
    return paths


def assign_hands(hands, left_wrist, right_wrist):
    """Assign detected hands to (left, right) slots.

    ``hands``: list of up to 2 arrays ``(21, 2)`` in 0-1 image coords.
    ``left_wrist`` / ``right_wrist``: pose wrists (2,), NaN if missing.
    Returns ``(left, right)``; either may be None. MediaPipe's handedness label
    is deliberately ignored. Without usable pose wrists, falls back to image x:
    in an un-mirrored image the person's left hand is on the larger-x side.
    """
    hands = list(hands)[:2]
    if not hands:
        return None, None
    wrists = np.array([h[0] for h in hands], dtype=np.float32)
    targets = [np.asarray(left_wrist, dtype=np.float32), np.asarray(right_wrist, dtype=np.float32)]
    valid = [bool(np.all(np.isfinite(t))) for t in targets]

    if not any(valid):
        if len(hands) == 1:
            return (hands[0], None) if wrists[0, 0] > 0.5 else (None, hands[0])
        left_i = int(np.argmax(wrists[:, 0]))
        return hands[left_i], hands[1 - left_i]

    def cost(hand_i, slot):
        if not valid[slot]:
            return MISSING_WRIST_COST
        return float(np.linalg.norm(wrists[hand_i] - targets[slot]))

    if len(hands) == 1:
        return (hands[0], None) if cost(0, 0) <= cost(0, 1) else (None, hands[0])
    straight = cost(0, 0) + cost(1, 1)
    swapped = cost(1, 0) + cost(0, 1)
    return (hands[0], hands[1]) if straight <= swapped else (hands[1], hands[0])


class LandmarkExtractor:
    """Per-frame pose + hand landmarks via the MediaPipe Tasks API (VIDEO mode)."""

    def __init__(self, model_dir="models"):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        pose_path, hand_path = download_models(model_dir)
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=pose_path),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=hand_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            )
        )
        self._t = 0.0
        self._last_ts = 0

    def _timestamp(self, dt_ms):
        self._t += dt_ms
        ts = max(int(self._t), self._last_ts + 1)  # strictly increasing across all videos
        self._last_ts = ts
        return ts

    def process(self, frame_bgr, dt_ms=33.3):
        """Return a ``(61, 2)`` float32 array for one BGR frame (NaN = missing)."""
        import cv2

        h, w = frame_bgr.shape[:2]
        if w > MAX_WIDTH:
            frame_bgr = cv2.resize(frame_bgr, (MAX_WIDTH, int(round(h * MAX_WIDTH / w))))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        ts = self._timestamp(dt_ms)

        out = np.full((61, 2), np.nan, dtype=np.float32)
        pose = self.pose.detect_for_video(image, ts)
        left_wrist = right_wrist = np.full(2, np.nan, dtype=np.float32)
        if pose.pose_landmarks:
            lm = pose.pose_landmarks[0]
            pts = np.array([[lm[i].x, lm[i].y] for i in POSE_IDX], dtype=np.float32)
            out[:19] = pts
            left_wrist, right_wrist = pts[15], pts[16]

        res = self.hands.detect_for_video(image, ts)
        hands = [np.array([[p.x, p.y] for p in hl], dtype=np.float32) for hl in res.hand_landmarks]
        left, right = assign_hands(hands, left_wrist, right_wrist)
        if left is not None:
            out[19:40] = left
        if right is not None:
            out[40:61] = right
        return out

    def new_video(self):
        self._t += 1000.0  # gap so trackers do not see a continuous stream

    def close(self):
        self.pose.close()
        self.hands.close()


def clean_label(name):
    return re.sub(r"^\s*\d+\s*[.\-_)]*\s*", "", name).strip().lower()


def signer_of(rel_path, regex):
    if regex is None:
        return "unknown"
    m = re.search(regex, rel_path)
    return m.group(1) if m and m.groups() else "unknown"


def find_videos(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def norm_cat(name):
    return re.sub(r"[\s_\-]+", "", name).lower()


def entries_from_csv(csv_path, category=None):
    """Rows of the split CSV as entries; ids are CSV row numbers so resume works across runs."""
    out = []
    with open(csv_path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if category and norm_cat(r["video_path"].split("/")[0]) != norm_cat(category):
                continue
            out.append(dict(id=i, video_path=r["video_path"], label=r["label"],
                            split=r["split"], signer="unknown"))
    return out


def entries_from_none(none_dir, video_root, first_id):
    """Non-sign clips: label 'none', every 5th clip (sorted) goes to test, the rest to train."""
    out = []
    for k, v in enumerate(find_videos(none_dir)):
        rel = Path(os.path.relpath(v, video_root)).as_posix()
        out.append(dict(id=first_id + k, video_path=rel, label="none",
                        split="test" if k % 5 == 4 else "train", signer="unknown"))
    return out


def entries_from_folder(root, signer_regex):
    out = []
    for i, v in enumerate(find_videos(root)):
        rel = v.relative_to(root).as_posix()
        out.append(dict(id=i, video_path=rel, label=clean_label(v.parent.name),
                        split="", signer=signer_of(rel, signer_regex)))
    return out


def extract_video(extractor, path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1000.0 / fps if fps and fps > 1 else 33.3
    extractor.new_video()
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(extractor.process(frame, dt))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 61, 2), np.float32)


_EXTRACTOR = None


def _init_worker(models_dir):
    global _EXTRACTOR
    _EXTRACTOR = LandmarkExtractor(models_dir)


def _work(job):
    """Process one video in a worker; returns (id, status). Writes the .npy itself."""
    vid, path, dest = job
    try:
        arr = extract_video(_EXTRACTOR, path)
    except Exception as e:  # keep the pool alive on a corrupt video
        return vid, f"error: {e}"
    if len(arr) == 0 or not np.any(np.isfinite(arr[:, 19:, 0])):
        return vid, "nohand"
    np.save(dest, arr.astype(np.float32))
    return vid, "ok"


def write_index(entries, out):
    """Rebuild index.csv from every entry whose .npy exists (accumulates across runs)."""
    rows = []
    for e in entries:
        p = out / "landmarks" / f"{e['id']}.npy"
        if p.exists():
            frames = np.load(p, mmap_mode="r").shape[0]
            rows.append([e["id"], e["video_path"], e["label"], e["split"], e["signer"], frames])
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "video_path", "label", "split", "signer", "frames"])
        w.writerows(rows)
    return len(rows)


def run(entries, video_root, out, workers=2, models_dir="models"):
    import multiprocessing as mp

    out = Path(out)
    (out / "landmarks").mkdir(parents=True, exist_ok=True)
    skipped_file = out / "skipped.txt"
    known_skips = set(skipped_file.read_text().splitlines()) if skipped_file.exists() else set()
    jobs, missing, done = [], 0, 0
    for e in entries:
        dest = out / "landmarks" / f"{e['id']}.npy"
        src = Path(video_root) / e["video_path"]
        if dest.exists() or e["video_path"] in known_skips:
            done += 1
        elif not src.exists():
            missing += 1
        else:
            jobs.append((e["id"], str(src), str(dest)))
    print(f"{len(entries)} entries: {done} already done, {missing} videos not present, {len(jobs)} to process")
    paths = {e["id"]: e["video_path"] for e in entries}
    if jobs:
        # Fail fast with the real error: a Pool whose initializer keeps crashing just respawns forever.
        LandmarkExtractor(models_dir).close()
        new_skips = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, _init_worker, (models_dir,)) as pool:
            for n, (vid, status) in enumerate(pool.imap_unordered(_work, jobs), 1):
                if status != "ok":
                    print(f"SKIP ({status}): {paths[vid]}")
                    if status == "nohand":
                        new_skips.append(paths[vid])
                if n % 25 == 0:
                    print(f"{n}/{len(jobs)}")
        if new_skips:
            with open(skipped_file, "a") as f:
                f.write("\n".join(new_skips) + "\n")
    return write_index(entries, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", help="folder mode: root folder of videos (label = parent folder)")
    ap.add_argument("--csv", help="split CSV (video_path, label, split), e.g. data/include50.csv")
    ap.add_argument("--video-root", default="data/include/", help="CSV mode: videos live under this folder")
    ap.add_argument("--category", help="CSV mode: only this category (first path component), e.g. Greetings")
    ap.add_argument("--none-dir", help="optional folder of non-sign clips (label 'none')")
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--signer-regex", default=None, help="folder mode: group 1 on the relative path = signer ID")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    if args.csv:
        entries = entries_from_csv(args.csv, args.category)
        if args.none_dir and (not args.category or norm_cat(args.category) == "none"):
            first = sum(1 for _ in open(args.csv)) - 1
            entries += entries_from_none(args.none_dir, args.video_root, first)
        root = args.video_root
        # index covers the full CSV so rows from earlier category runs are kept
        index_entries = entries_from_csv(args.csv)
        if args.none_dir:
            index_entries += entries_from_none(args.none_dir, args.video_root, len(index_entries))
    elif args.root:
        entries = index_entries = entries_from_folder(args.root, args.signer_regex)
        root = args.root
    else:
        ap.error("give a root folder or --csv")
    run(entries, root, args.out, args.workers, args.models_dir)
    n = write_index(index_entries, Path(args.out))
    print(f"index.csv: {n} clips in {args.out}")


if __name__ == "__main__":
    main()
