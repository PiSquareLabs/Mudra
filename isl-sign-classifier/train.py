"""Signer-split training with per-class report."""
import argparse
import csv
import json
import math
import os
from collections import Counter

import numpy as np

import model as M  # noqa: F401  (registers custom layers)
import keras
from sklearn.model_selection import GroupShuffleSplit, train_test_split


class SignSequence(keras.utils.PyDataset):
    def __init__(self, clips, labels, n_classes, batch, train, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.clips, self.labels, self.n_classes = clips, np.asarray(labels), n_classes
        self.batch, self.train = batch, train
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(clips))
        if train:
            self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.clips) / self.batch)

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        f = (lambda c: M.augment(c, self.rng)) if self.train else M.prepare_val
        x = np.stack([f(self.clips[j]) for j in idx])
        y = keras.utils.to_categorical(self.labels[idx], self.n_classes)
        return x, y

    def on_epoch_end(self):
        if self.train:
            self.rng.shuffle(self.order)


def load_index(data):
    with open(os.path.join(data, "index.csv")) as f:
        return list(csv.DictReader(f))


def official_split(rows):
    """Use the CSV's own train / val / test column (val picks the checkpoint, test is for the report)."""
    cols = np.array([r["split"] for r in rows])
    parts = {k: np.nonzero(cols == k)[0] for k in ("train", "val", "test")}
    if not len(parts["train"]) or not len(parts["val"]):
        raise SystemExit("--split-col needs rows labelled 'train' and 'val'; extract more categories first.")
    return parts["train"], parts["val"], parts["test"]


def split(rows, val_signers, seed):
    signers = np.array([r["signer"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    idx = np.arange(len(rows))
    if val_signers:
        val_mask = np.isin(signers, val_signers)
        return idx[~val_mask], idx[val_mask]
    known = set(signers) - {"unknown"}
    if len(known) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        return next(gss.split(idx, groups=signers))
    print("WARNING: no signer information (fewer than 2 known signers). Using a stratified split "
          "by video, so validation accuracy will be too optimistic (same signers in train and val). "
          "Pass --signer-regex to extract_landmarks.py for a real estimate.")
    try:
        return train_test_split(idx, test_size=0.2, stratify=labels, random_state=seed)
    except ValueError:
        return train_test_split(idx, test_size=0.2, random_state=seed)


def report(model, val_seq, y_true, labels):
    probs = model.predict(val_seq, verbose=0)
    pred = probs.argmax(1)
    print(f"\nOverall val accuracy: {(pred == y_true).mean():.4f} ({len(y_true)} clips)")
    print("\nPer-class accuracy:")
    for c, name in enumerate(labels):
        m = y_true == c
        if m.any():
            print(f"  {name:<30} {(pred[m] == c).mean():.3f}  (n={m.sum()})")
    conf = Counter((labels[t], labels[p]) for t, p in zip(y_true, pred) if t != p)
    print("\nTop 10 confusions (true -> predicted):")
    for (t, p), n in conf.most_common(10):
        print(f"  {t} -> {p}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out", help="folder with index.csv and landmarks/")
    ap.add_argument("--out", default=None, help="where to write model.keras / labels.json (default: --data)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--max-classes", type=int, default=0)
    ap.add_argument("--val-signers", default="")
    ap.add_argument("--split-col", default=None,
                    help="index.csv column with train/val/test (use 'split' for the official INCLUDE-50 split)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out or args.data
    os.makedirs(out, exist_ok=True)
    keras.utils.set_random_seed(args.seed)

    rows = load_index(args.data)
    if args.max_classes:
        top = {l for l, _ in Counter(r["label"] for r in rows).most_common(args.max_classes)}
        rows = [r for r in rows if r["label"] in top]
    labels = sorted({r["label"] for r in rows})
    lid = {l: i for i, l in enumerate(labels)}
    y = np.array([lid[r["label"]] for r in rows])
    clips = [np.load(os.path.join(args.data, "landmarks", f"{r['id']}.npy")) for r in rows]
    print(f"{len(rows)} clips, {len(labels)} classes")

    te = np.array([], dtype=int)
    if args.split_col:
        for r in rows:
            r["split"] = r[args.split_col]
        tr, va, te = official_split(rows)
    else:
        val_signers = [s for s in args.val_signers.split(",") if s]
        tr, va = split(rows, val_signers, args.seed)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)}")
    train_seq = SignSequence([clips[i] for i in tr], y[tr], len(labels), args.batch, True, args.seed)
    val_seq = SignSequence([clips[i] for i in va], y[va], len(labels), args.batch, False)

    model = M.build_model(len(labels), args.dim, args.blocks)
    steps = max(1, len(train_seq)) * args.epochs
    model.compile(
        optimizer=keras.optimizers.AdamW(keras.optimizers.schedules.CosineDecay(1e-3, steps), weight_decay=0.05),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )
    model_path = os.path.join(out, "model.keras")
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    model.fit(
        train_seq, validation_data=val_seq, epochs=args.epochs, verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy",
                                                   mode="max", save_best_only=True)],
    )
    best = keras.models.load_model(model_path)
    if len(te):
        print("\n=== Final report on the official TEST split ===")
        test_seq = SignSequence([clips[i] for i in te], y[te], len(labels), args.batch, False)
        report(best, test_seq, y[te], labels)
    else:
        report(best, val_seq, y[va], labels)


if __name__ == "__main__":
    main()
