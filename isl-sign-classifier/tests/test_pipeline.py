import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import keras  # noqa: E402
import model as M  # noqa: E402
from extract_landmarks import assign_hands, clean_label, signer_of  # noqa: E402
from live_demo import SignSegmenter  # noqa: E402


def clip(n=40, seed=0):
    return np.random.default_rng(seed).uniform(0.2, 0.8, (n, M.N_PTS, 2)).astype(np.float32)


def run_pre(x):
    return M.Preprocess()(x).numpy()


def test_preprocess_finite_with_missing_data():
    x = np.stack([clip(M.T, i) for i in range(4)])
    x[0, :, M.LEFT] = np.nan  # left hand missing
    x[1, :, M.RIGHT] = np.nan  # right hand missing
    x[2, :, :M.N_POSE] = np.nan  # no pose
    x[3] = np.nan  # nothing at all
    x[0, 5:9, 11] = np.nan  # shoulders missing in some frames
    out = run_pre(x)
    assert out.shape == (4, M.T, M.N_FEATURES)
    assert np.isfinite(out).all()
    assert np.abs(out).max() <= 10
    assert out[0, :, -2].sum() == 0 and out[0, :, -1].sum() == M.T  # presence flags


def test_mirror_twice_is_identity():
    x = clip()
    x[3, M.LEFT] = np.nan
    twice = M.mirror(M.mirror(x))
    np.testing.assert_allclose(twice, x, atol=1e-6, equal_nan=True)
    once = M.mirror(x)
    np.testing.assert_allclose(once[:, M.RIGHT, 0], 1 - x[:, M.LEFT, 0], atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(once[:, 1], np.stack([1 - x[:, 4, 0], x[:, 4, 1]], -1), atol=1e-6)


def test_augment_and_val_shapes():
    rng = np.random.default_rng(0)
    assert M.augment(clip(50), rng).shape == (M.T, M.N_PTS, 2)
    assert M.prepare_val(clip(10)).shape == (M.T, M.N_PTS, 2)


def hand_at(p):
    return np.tile(np.array(p, np.float32), (21, 1)) + np.random.default_rng(1).normal(0, 0.01, (21, 2)).astype(np.float32)


def test_hand_assignment_with_pose():
    lw, rw = np.array([0.7, 0.5], np.float32), np.array([0.3, 0.5], np.float32)
    near_l, near_r = hand_at([0.68, 0.52]), hand_at([0.31, 0.48])
    left, right = assign_hands([near_r, near_l], lw, rw)  # detection order must not matter
    assert left is near_l and right is near_r
    left, right = assign_hands([near_r], lw, rw)
    assert left is None and right is near_r
    nan = np.full(2, np.nan, np.float32)
    left, right = assign_hands([near_r, near_l], nan, nan)  # image-x fallback
    assert left is near_l and right is near_r


def test_label_and_signer():
    assert clean_label("12. Hello") == "hello"
    assert clean_label("7 Thank You") == "thank you"
    assert signer_of("a/b/user_3_x.mp4", r"user_(\d+)") == "3"
    assert signer_of("a/b.mp4", None) == "unknown"


def test_segmenter():
    hand, none = clip(1)[0], clip(1)[0].copy()
    none[19:] = np.nan
    seg = SignSegmenter()
    outs = [seg.push(none) for _ in range(5)] + [seg.push(hand) for _ in range(12)] + [seg.push(none) for _ in range(8)]
    got = [o for o in outs if o is not None]
    assert len(got) == 1 and len(got[0]) == 12
    short = [seg.push(hand) for _ in range(3)] + [seg.push(none) for _ in range(8)]
    assert all(o is None for o in short)


def test_tiny_model_train_save_export(tmp_path):
    import export_tflite as E

    rng = np.random.default_rng(0)
    n_cls = 3
    labels = np.repeat(np.arange(n_cls), 16)
    # class-dependent offset so the tiny model learns something decisive
    x = np.stack([M.prepare_val(clip(36, i) + 0.1 * labels[i]) for i in range(len(labels))])
    y = keras.utils.to_categorical(labels, n_cls)
    model = M.build_model(n_cls, dim=32, blocks=2)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.fit(x, y, epochs=2, batch_size=16, verbose=0)
    path = str(tmp_path / "model.keras")
    model.save(path)
    reloaded = keras.models.load_model(path)
    xs = E.random_inputs(5)
    np.testing.assert_allclose(model.predict(xs[:, 0], verbose=0), reloaded.predict(xs[:, 0], verbose=0), atol=1e-5)

    paths = E.export(reloaded, str(tmp_path))
    assert E.verify(reloaded, paths, n=20)
