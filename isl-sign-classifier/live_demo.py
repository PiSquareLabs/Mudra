"""Webcam test: segment signs by hand presence, classify with the .tflite file."""
import argparse
import json
import time

import numpy as np

import model as M
import tensorflow as tf
from extract_landmarks import LandmarkExtractor

MIN_FRAMES = 8
END_AFTER = 8
THRESHOLD = 0.6


class SignSegmenter:
    """Start when a hand appears; end after END_AFTER hand-less frames; drop short segments."""

    def __init__(self, end_after=END_AFTER, min_frames=MIN_FRAMES):
        self.end_after, self.min_frames = end_after, min_frames
        self.buf, self.misses = [], 0

    @property
    def active(self):
        return bool(self.buf)

    def push(self, frame):
        """Feed one ``(61, 2)`` frame; returns a ``(n, 61, 2)`` segment when one ends."""
        has_hand = bool(np.any(np.isfinite(frame[19:, 0])))
        if not self.buf:
            if has_hand:
                self.buf, self.misses = [frame], 0
            return None
        self.buf.append(frame)
        self.misses = 0 if has_hand else self.misses + 1
        if self.misses < self.end_after:
            return None
        seg = np.stack(self.buf[:-self.misses])
        self.buf, self.misses = [], 0
        return seg if len(seg) >= self.min_frames else None


class Classifier:
    def __init__(self, tflite_path, labels):
        self.it = tf.lite.Interpreter(model_path=tflite_path)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.labels = labels

    def predict(self, segment):
        x = M.prepare_val(segment)[None]
        self.it.set_tensor(self.inp, x)
        self.it.invoke()
        p = self.it.get_tensor(self.out)[0]
        i = int(p.argmax())
        return self.labels[i], float(p[i])


def draw(frame, pts, text):
    import cv2

    h, w = frame.shape[:2]
    for k, (x, y) in enumerate(pts):
        if np.isfinite(x):
            color = (0, 200, 255) if k < 19 else (0, 255, 0) if k < 40 else (255, 120, 0)
            cv2.circle(frame, (int(x * w), int(y * h)), 3, color, -1)
    if text:
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="out/model_fp16.tflite")
    ap.add_argument("--labels", default="out/labels.json")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    clf = Classifier(args.model, json.load(open(args.labels)))
    ext = LandmarkExtractor(args.models_dir)
    seg = SignSegmenter()
    cap = cv2.VideoCapture(args.camera)
    shown, shown_until, last = "", 0.0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        pts = ext.process(frame, (now - last) * 1000.0)
        last = now
        done = seg.push(pts)
        if done is not None:
            label, prob = clf.predict(done)
            if prob > THRESHOLD:
                shown, shown_until = f"{label} ({prob:.0%})", now + 2.0
        text = shown if now < shown_until else ("..." if seg.active else "")
        draw(frame, pts, text)
        cv2.imshow("ISL demo (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    ext.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
