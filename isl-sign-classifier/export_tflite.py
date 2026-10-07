"""Export model.keras to float16 and dynamic-range INT8 TFLite, then verify."""
import argparse
import os
import sys

import numpy as np

import model as M  # noqa: F401  (registers custom layers)
import keras
import tensorflow as tf


def _concrete(model):
    """Fixed-shape concrete function with variables frozen into constants.

    Keras 3 variables are not tracked by the converter, so without freezing the
    TFLite graph would read uninitialised resource variables.
    """
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    @tf.function(input_signature=[tf.TensorSpec([1, M.T, M.N_PTS, 2], tf.float32)])
    def serve(x):
        return model(x, training=False)

    return convert_variables_to_constants_v2(serve.get_concrete_function())


def export(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    conc = _concrete(model)
    paths = {}
    for name, fp16 in (("fp16", True), ("int8", False)):
        conv = tf.lite.TFLiteConverter.from_concrete_functions([conc])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if fp16:
            conv.target_spec.supported_types = [tf.float16]
        path = os.path.join(out_dir, f"model_{name}.tflite")
        with open(path, "wb") as f:
            f.write(conv.convert())
        paths[name] = path
        print(f"{path}: {os.path.getsize(path) / 1024:.1f} KiB")
    return paths


def random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 0.8, size=(n, 1, M.T, M.N_PTS, 2)).astype(np.float32)
    for i in range(n):
        for sl in (M.LEFT, M.RIGHT):
            if rng.random() < 0.3:  # whole hand missing in some frames
                x[i, 0, rng.random(M.T) < 0.4, sl] = np.nan
        mask = rng.random((M.T, M.N_PTS)) < 0.05  # scattered missing points
        mask[:, 11:13] = False
        x[i, 0][mask] = np.nan
    x[0, 0, :, :M.N_POSE] = np.nan  # a clip with no pose at all
    return x


def run_tflite(path, xs):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = []
    for x in xs:
        it.set_tensor(inp["index"], x)
        it.invoke()
        res.append(it.get_tensor(out["index"])[0].copy())
    return np.stack(res)


def verify(model, paths, n=20, min_agree=0.95, seed=0):
    xs = random_inputs(n, seed)
    ref = np.stack([model(x, training=False).numpy()[0] for x in xs])
    ok = True
    for name, path in paths.items():
        got = run_tflite(path, xs)
        diff = float(np.abs(ref - got).max())
        agree = float((ref.argmax(1) == got.argmax(1)).mean())
        print(f"{name}: max |diff| = {diff:.5f}, top-1 agreement = {agree:.0%}")
        ok &= agree >= min_agree
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/model.keras")
    ap.add_argument("--out", default=None, help="output folder (default: next to the model)")
    ap.add_argument("--n-checks", type=int, default=20)
    args = ap.parse_args()
    model = keras.models.load_model(args.model)
    paths = export(model, args.out or os.path.dirname(os.path.abspath(args.model)))
    if not verify(model, paths, args.n_checks):
        sys.exit("FAILED: top-1 agreement below 95%")
    print("Verification passed.")


if __name__ == "__main__":
    main()
