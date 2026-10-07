"""Constants, preprocessing layer, ECA layer, model builder and NumPy augmentation.

All feature engineering lives inside the Keras model (``Preprocess``), so the
exported TFLite file consumes raw landmarks of shape ``[1, 32, 61, 2]``.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402

T = 32  # frames per clip
N_PTS = 61  # 19 pose + 21 left hand + 21 right hand
N_POSE = 19
LEFT = slice(19, 40)
RIGHT = slice(40, 61)
# MediaPipe pose indices stored in rows 0..18
POSE_IDX = list(range(17)) + [23, 24]
# swap pairs inside the 19-row pose block (mirror augmentation)
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18)]
N_FEATURES = 122 + 84 + 122 + 2  # body, hand-local, velocity, presence


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package="isl")
class Preprocess(layers.Layer):
    """Raw ``(B, T, 61, 2)`` coordinates (NaN = missing) -> ``(B, T, 330)`` features."""

    def call(self, x):
        x = tf.cast(x, tf.float32)
        t = x.shape[1]
        isnan = tf.math.is_nan

        # --- body-normalised coordinates -----------------------------------
        ls, rs = x[:, :, 11, :], x[:, :, 12, :]  # (B, T, 2)
        bad = tf.logical_or(isnan(ls), isnan(rs))
        valid = tf.logical_not(tf.logical_or(bad[:, :, 0], bad[:, :, 1]))  # (B, T)
        w = tf.cast(valid, tf.float32)
        denom = tf.maximum(tf.reduce_sum(w, axis=1), 1.0)  # (B,)
        has = tf.reduce_sum(w, axis=1) > 0.0

        mid = tf.where(valid[:, :, None], (ls + rs) * 0.5, 0.0)
        center = tf.reduce_sum(mid * w[:, :, None], axis=1) / denom[:, None]  # (B, 2)
        center = tf.where(has[:, None], center, 0.5)  # no pose at all: image centre
        diff = tf.where(valid[:, :, None], ls - rs, 0.0)
        width = tf.sqrt(tf.reduce_sum(diff * diff, axis=-1))  # (B, T)
        scale = tf.reduce_sum(width * w, axis=1) / denom + 1e-6
        scale = tf.where(has, scale, 1.0)  # (B,)

        body = (x - center[:, None, None, :]) / scale[:, None, None, None]
        body_flat = tf.reshape(body, [-1, t, N_PTS * 2])

        # --- hand-local shape -------------------------------------------------
        local, pres = [], []
        for sl in (LEFT, RIGHT):
            hand = x[:, :, sl, :]  # (B, T, 21, 2)
            wrist = hand[:, :, 0:1, :]
            ref = hand[:, :, 9, :] - hand[:, :, 0, :]
            ref = tf.sqrt(tf.reduce_sum(ref * ref, axis=-1)) + 1e-6  # (B, T)
            loc = (hand - wrist) / ref[:, :, None, None]
            local.append(tf.reshape(loc, [-1, t, 42]))
            pres.append(tf.cast(tf.logical_not(isnan(wrist[:, :, 0, 0])), tf.float32)[:, :, None])

        # --- velocity ------------------------------------------------------------
        vel = tf.concat([tf.zeros_like(body_flat[:, :1]), body_flat[:, 1:] - body_flat[:, :-1]], axis=1)

        feats = tf.concat([body_flat, local[0], local[1], vel, pres[0], pres[1]], axis=-1)
        feats = tf.where(isnan(feats), 0.0, feats)
        return tf.clip_by_value(feats, -10.0, 10.0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], N_FEATURES)


@keras.saving.register_keras_serializable(package="isl")
class ECA(layers.Layer):
    """Efficient channel attention: time-mean -> Conv1D across channels -> sigmoid -> scale."""

    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv1D(1, kernel_size, padding="same", use_bias=False)

    def build(self, input_shape):
        self.conv.build((None, input_shape[-1], 1))
        super().build(input_shape)

    def call(self, x):
        s = keras.ops.mean(x, axis=1, keepdims=True)  # (B, 1, C)
        s = keras.ops.transpose(s, (0, 2, 1))  # (B, C, 1)
        s = keras.ops.sigmoid(self.conv(s))
        return x * keras.ops.transpose(s, (0, 2, 1))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update(kernel_size=self.kernel_size)
        return cfg


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(n_classes, dim=192, blocks=6, kernel=11, verbose=True):
    inp = keras.Input((T, N_PTS, 2), name="landmarks")
    x = Preprocess(name="preprocess")(inp)
    x = layers.Dense(dim)(x)
    x = layers.BatchNormalization()(x)
    for _ in range(blocks):
        r = x
        x = layers.Dense(2 * dim, activation="swish")(x)
        x = layers.ZeroPadding1D((kernel - 1, 0))(x)  # causal
        x = layers.DepthwiseConv1D(kernel, padding="valid")(x)
        x = layers.BatchNormalization()(x)
        x = ECA()(x)
        x = layers.Dense(dim)(x)
        x = layers.Dropout(0.2, noise_shape=(None, 1, 1))(x)
        x = layers.Add()([r, x])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inp, out, name="isl_sign_classifier")
    if verbose:
        print(f"Parameters: {model.count_params():,}")
    return model


# --------------------------------------------------------------------------- #
# Augmentation (NumPy, raw coordinates)
# --------------------------------------------------------------------------- #
def resample(seq, n_out=T, start=0, length=None):
    """Nearest-index resample of ``seq[start:start+length]`` to ``n_out`` frames."""
    n = len(seq)
    length = n - start if length is None else length
    idx = np.rint(np.linspace(start, start + length - 1, n_out)).astype(int)
    return seq[np.clip(idx, 0, n - 1)]


def time_crop(seq, rng, lo=0.8, hi=1.0):
    n = len(seq)
    length = max(1, int(round(n * rng.uniform(lo, hi))))
    start = int(rng.integers(0, n - length + 1))
    return resample(seq, T, start, length)


def mirror(seq):
    """Horizontal flip: x -> 1 - x, swap hand blocks and left/right pose pairs."""
    out = seq.copy()
    out[:, LEFT] = seq[:, RIGHT]
    out[:, RIGHT] = seq[:, LEFT]
    for a, b in POSE_SWAP:
        out[:, a] = seq[:, b]
        out[:, b] = seq[:, a]
    out[..., 0] = 1.0 - out[..., 0]
    return out


def geometric(seq, rng, max_rot=15.0, scale=(0.8, 1.2), shift=0.1):
    """Random rotation / scale / shift of the whole clip around (0.5, 0.5)."""
    a = np.deg2rad(rng.uniform(-max_rot, max_rot))
    s = rng.uniform(*scale)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32) * s
    c = np.float32(0.5)
    out = (seq - c) @ rot.T + c + rng.uniform(-shift, shift, size=2).astype(np.float32)
    return out.astype(np.float32)


def hand_dropout(seq, rng, p=0.05):
    """Set each hand to NaN in ~p of the frames."""
    out = seq.copy()
    for sl in (LEFT, RIGHT):
        out[rng.random(len(out)) < p, sl] = np.nan
    return out


def augment(seq, rng):
    seq = time_crop(seq, rng)
    if rng.random() < 0.5:
        seq = mirror(seq)
    seq = geometric(seq, rng)
    return hand_dropout(seq, rng).astype(np.float32)


def prepare_val(seq):
    """Validation / inference clip: resample only."""
    return resample(seq, T).astype(np.float32)
