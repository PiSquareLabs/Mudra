# ISL sign classifier (INCLUDE / INCLUDE-50)

Trains a small isolated-sign classifier for Indian Sign Language from MediaPipe landmarks and
exports it to TFLite for a Flutter Android app. All preprocessing (normalisation, hand-local
shape, velocity) is inside the Keras model, so the app sends only raw landmarks.
An installable landing page (PWA) lives in [`web/`](web/).

## Setup

```bash
pip install -r requirements.txt     # Python 3.10+
pytest                               # synthetic data only, no videos or MediaPipe models needed
```

## Data: INCLUDE-50 (official split)

Split files are in `data/splits/` (AI4Bharat, CC BY 4.0). Turn them into a CSV:

```bash
python prepare_split.py     # -> data/include50.csv (video_path, label, split) + scripts/needed_categories.txt
```

958 videos (689 train / 77 val / 192 test), 50 labels, spread over all 15 category zips
(~57 GB in total). Download them from the Zenodo record <https://zenodo.org/records/4010759>
and unzip each into `data/include/`, so videos sit at `data/include/<Category>/<N. Sign>/<file>`.
(Full INCLUDE, ~263 signs: use the original folder mode below with `--signer-regex`.)

### One category at a time (disk friendly)

Landmark arrays are tiny compared to videos, so process each zip and delete the videos:

```bash
# for each category in scripts/needed_categories.txt:
unzip Greetings.zip -d data/include/
python extract_landmarks.py --csv data/include50.csv --video-root data/include/ --category Greetings
rm -rf data/include/Greetings Greetings.zip
```

- Only videos present on disk for that category are processed; missing ones are skipped silently.
- Resume: a video whose `out/landmarks/<id>.npy` exists (or that had no hand) is not redone.
  `<id>` is the CSV row number, so ids stay stable across categories.
- `out/index.csv` (`id, video_path, label, split, signer, frames`) is rebuilt after every run from
  all `.npy` files so far.
- `--workers N` sets the number of worker processes (one MediaPipe extractor each).
- Optional non-sign clips: `--none-dir data/none` adds label `none` (every 5th clip to test, rest train).

## Train, export, demo

```bash
python train.py --data out --split-col split      # official train / val / test
python export_tflite.py --model out/model.keras
python live_demo.py --model out/model_fp16.tflite --labels out/labels.json
```

With `--split-col split`, `train` fits the model, `val` selects the best checkpoint
(`val_accuracy`), and `test` is used once, for the final report (accuracy, per-class, top-10
confusions). Without it, `train.py` splits by signer (`--signer-regex` at extraction) or, if there are no
signers, stratified by video with a WARNING that accuracy is optimistic. Labels are the classes present in
`index.csv`, so train after extracting every category.

Other options: `--epochs 80 --batch 64 --dim 192 --blocks 6 --max-classes N --val-signers a,b --seed`.
Folder mode extraction (any dataset): `python extract_landmarks.py <root> --signer-regex "..."`,
label = parent folder name with a leading `12. ` removed, lowercased.

## Android input contract

One float32 input `[1, 32, 61, 2]`: 32 frames, 61 points, x/y in 0-1 image coordinates, `NaN` for
missing points. Rows: 0-18 pose (indices 0-16, 23, 24), 19-39 person's LEFT hand, 40-60 RIGHT hand.
Assign hands by distance from the hand wrist to the pose wrists (15 = left, 16 = right); do not
use MediaPipe's handedness label. Use the Tasks API (PoseLandmarker lite + HandLandmarker), resize
to max 640 px width. Output: softmax over `labels.json`. Models: `model_fp16.tflite`, `model_int8.tflite` (dynamic range).

## Choices made

- Hand fallback when no pose wrists exist: image x (un-mirrored image: larger x = person's left).
- A hand is put on a slot whose pose wrist is missing at a fixed cost of 1.0.
- No shoulders in any frame: centre 0.5 and scale 1 (instead of dividing by 1e-6).
- Export freezes Keras 3 variables into constants before conversion; otherwise the TFLite
  graph reads uninitialised variables and outputs NaN.
- ~1 M parameters (dim 192, 6 blocks, kernel 11). Export verification fails below 95% top-1 agreement.
- Landmarks use VIDEO mode, one timestamp stream per worker process.
- Live demo: a sign segment starts when a hand appears, ends after 8 hand-less frames
  (those frames are trimmed), segments under 8 frames are ignored, label shown above probability 0.6.
- Real training and real-video extraction were not run in development; only synthetic tests.
