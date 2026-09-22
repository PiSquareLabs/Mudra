<div align="center">

# MUDRA+

**An on-device ASL input accelerator — a small, high-accuracy sign vocabulary plus autocomplete,
so a spoken sentence costs a few signs instead of a whole sentence of them.**

Built and completed at **iQOO City Battles, Chennai**, on an **iQOO 15 (Snapdragon 8 Elite Gen 5)**.

</div>

```
sign 2–3 glosses  →  ME NEED METFORMIN  →  candidate sentences  →  you pick one  →  spoken
                                             ↑                       ↑
                            memory (~1 µs) or on-device 3B     nothing is spoken
                                                                before this point
```

Everything in the live path runs on the phone: hand tracking, sentence composition,
transcription, speech and memory. No server, no API key, no account.

---

## Contents

- [Scope](#scope)
- [Quick start](#quick-start)
- [Full setup guide](#full-setup-guide)
- [Design rationale](#design-rationale)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Recognition](#recognition)
- [Autocomplete](#autocomplete)
- [LLM layer](#llm-layer)
- [Audio and the relay](#audio-and-the-relay)
- [Models](#models)
- [On-device notes](#on-device-notes--snapdragon-8-elite-gen-5)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Status and limitations](#status-and-limitations)

---

## Scope

A prototype, with a deliberately narrow scope:

- It is an **autocomplete for a signer's own recurring phrases**, not a translation system. The
  system has no opinion about grammar it has not been shown; it proposes candidate sentences and
  the person chooses one.
- **Interpretation is a human skill and this does not attempt it.** The design assumes someone
  driving a tool for their own phrases, in their own words, with a confirmation step they
  control.
- The demo vocabulary is **small on purpose** — a handful of bundled single-hand static poses plus
  whatever the user records. The interesting question is not vocabulary size; it is how far a
  *reliable* vocabulary gets you when autocomplete carries the rest.
- Detection for the demo is **MediaPipe hand landmarks + geometric template matching**, not a
  trained sign classifier.
- The natural next step is evaluation **with signers**, on the ergonomics of the capture gate, the
  hold duration and whether the candidate sentences are worth choosing between. Every number below
  is an engineering measurement, not a user-study result.

---

## Quick start

For a machine that already has the Android toolchain:

```bash
git clone https://github.com/xreedev/mudra.git
cd mudra/app
npm install

# one-time: put the LLM weights on the device (see below for where to get them)
adb push qwen2.5-3b-instruct-q4_k_m.gguf /sdcard/Android/data/com.mudraapp/files/models/

npm run android          # device with USB debugging, or an emulator (no camera)
```

The hand-tracking model and the Whisper weights are already in the repo — only the `.gguf` has to
be fetched, because it is ~2 GB.

---

## Full setup guide

### 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Node | ≥ 18 (20 or 22 recommended) | `package.json` engines |
| JDK | 17 | Android Gradle Plugin 8.x |
| Android SDK | compileSdk 35, build-tools 35 | via Android Studio or `sdkmanager` |
| Android NDK | 26.1.10909125 | pinned in `android/build.gradle`; needed by VisionCamera and llama.rn |
| A physical Android device | Android 7.0+ (minSdk 24) | **required** — the camera path cannot be demoed on an emulator |
| Xcode + CocoaPods | only for iOS | the demo was built and tested on Android |

Set `ANDROID_HOME` and put `adb` on your `PATH`. Verify with `adb devices` — the device must show
as `device`, not `unauthorized`.

### 2. Install

```bash
git clone https://github.com/xreedev/mudra.git
cd mudra/app
npm install
```

### 3. Model assets

Three models. Two ship with the repo; one you supply.

| Model | Where it lives | Action |
|---|---|---|
| **MediaPipe hand landmarker** | `app/android/app/src/main/assets/hand_landmarker.task` | none — committed |
| **Whisper tiny** | `app/assets/models/ggml-tiny.bin` | none — committed (~75 MB) |
| **Qwen2.5-3B-Instruct Q4_K_M** | *not committed* (`.gguf` is gitignored; ~2 GB) | download and install, below |

Download `qwen2.5-3b-instruct-q4_k_m.gguf` from Hugging Face (for example the
`Qwen/Qwen2.5-3B-Instruct-GGUF` or `bartowski/Qwen2.5-3B-Instruct-GGUF` repositories), then choose
one of the two installation routes:

**Debug builds — push it to the device:**

```bash
adb shell mkdir -p /sdcard/Android/data/com.mudraapp/files/models
adb push qwen2.5-3b-instruct-q4_k_m.gguf /sdcard/Android/data/com.mudraapp/files/models/
```

That path is the app's own external files directory, so no storage permission is involved. The
filename must match `MODEL_FILENAME` in `app/src/llm/useLocalLlm.ts`. If the app starts without
finding it, it tells you the exact `adb push` command on screen.

**Release builds — bundle it into the APK:**

Copy the `.gguf` to `app/android/app/src/main/assets/models/` under the same filename; on first
launch the app extracts it to its runtime path. On Windows,
`app/scripts/build-release-with-llm.ps1` automates this (it also works around a `MAX_PATH` issue
in VisionCamera's release CMake build and an OOM in AGP's asset compression for multi-GB assets):

```powershell
.\scripts\build-release-with-llm.ps1 -ModelPath C:\models\qwen2.5-3b-instruct-q4_k_m.gguf
```

Model choice is not fixed — `llm-testbed/` is a standalone harness for benchmarking GGUFs on the
actual phone, with a table of sensible choices per RAM tier (1.5B at 6 GB, 3B at 8 GB, 7B at
12 GB+).

### 4. Run

```bash
npm start                 # Metro, in one terminal
npm run android           # build + install, in another
```

Grant **camera** and **microphone** when prompted — signing needs the first, the receive side
needs the second. Nothing is recorded or uploaded either way.

First launch does two slow things once: extracting the model (release builds) and loading it into
llama.cpp. After that it stays warm.

### 5. Try the flow

1. **Add custom sign** → hold a sign inside the guide circle until the ring completes → name it →
   save. It is usable on the next frame.
2. **Call** → hold signs to build a gloss sequence → pick one of the proposed sentences → it is
   spoken.
3. **Memory** → the sentence you picked is remembered for that sign sequence, and resurfaces next
   time instead of asking again.

### 6. Two-phone relay (optional)

The relay lets a second phone speak for the signer and stream a transcript back.

- Both phones on the **same Wi-Fi network**.
- Install the app on both; one opens **Call** (sender), the other opens **Receive**.
- Discovery is zeroconf/mDNS with service type `aslrelay`; the session runs over TCP on port
  **12345**. Some networks (guest Wi-Fi, client isolation) block mDNS or peer-to-peer traffic — a
  phone hotspot is the reliable fallback.

Without a relay, the app is still fully usable on one phone: put the call on speakerphone and let
TTS reach the other party acoustically (see `asl-call-relay.md` for why Android leaves no better
option).

### 7. The standalone modules

Each builds and tests on a laptop with no emulator, no device and no network:

```bash
cd memory-layer     && ./gradlew :core:test      # Kotlin engine; no Android SDK needed
cd memory-layer-rn  && npm install && npm test   # TypeScript port
cd sign-embeddings  && npm install && npm test   # embedding-based custom signs
cd llm-testbed      && npm install && npm run android   # GGUF benchmark harness (needs a device)
```

---

## Design rationale

### Why autocomplete instead of sign-every-word

Signing a full English sentence is expensive twice over: physically, and statistically.

If per-sign recognition accuracy is `p`, the chance an `n`-sign sentence survives end to end is
roughly `p^n`. The exponent is the problem:

| Signs needed | p = 0.95 | p = 0.90 |
|---|---|---|
| 8 (full sentence) | 0.66 | 0.43 |
| 5 | 0.77 | 0.59 |
| **3 (gloss key + autocomplete)** | **0.86** | **0.73** |

*(A model of the failure mode, not a measurement — but it is why the architecture is shaped this
way.)*

Two levers follow:

1. **Reduce `n`** — don't sign the sentence, sign the key. `ME NEED METFORMIN` is enough to
   retrieve or generate "I need one strip of Metformin, please."
2. **Raise `p`** — a small set of distinct, well-separated poses is far easier to get right than a
   large one full of near-collisions, and a rejection threshold means the system answers `UNKNOWN`
   instead of guessing.

Both point away from a large classifier and toward a small matcher plus strong autocomplete.

### Why no trained classifier in the live path

A classifier's output layer is fixed to the vocabulary it was trained on. Adding one sign means
collecting data, retraining, re-exporting and shipping a new model file — incompatible with the
thing this demo most wants to test: **a user adding their own sign in a second and using it on the
next frame.** So the live path is geometric:

```
21 landmarks → normalise around the wrist → 72-dim feature vector → RMS distance to templates
                                                                  → nearest, or UNKNOWN above τ
```

`sign-embeddings/` implements the fallback (reuse an internal layer of a trained TFLite model as an
embedding extractor, match by cosine similarity) for when template matching runs out of headroom.
Built and tested, not wired in, because the demo does not need it.

### Why the LLM is small and rarely called

Glosses are not English — no articles, tense or register. That is a language task, so a language
model does it rather than the vision path. But the LLM is the slowest and hottest component, so
the architecture minimises **how often** it runs:

- an exact sequence already confirmed → **memory, O(1), no model**;
- a sequence whose candidates the user has chosen from before → **sentence memory, no model**;
- only a genuinely new sequence reaches the 3B.

That ordering is also what keeps the app interactive under sustained camera load.

---

## Architecture

```
┌──────────────── PHONE (React Native 0.76 + native modules) ─────────────────────┐
│                                                                                  │
│  react-native-vision-camera                                                      │
│      │ frames                                                                    │
│      ▼                                                                           │
│  HandLandmarksFrameProcessorPlugin  (native, MediaPipe, hand_landmarker.task)    │
│      │ 21 × {x,y,z}                                                              │
│      ▼                                                                           │
│  assessCapture      hand present? close enough (handSpread)? steady (maxJitter)? │
│      ▼                                                                           │
│  gestureRecognizer  normalise → 72 features → RMS distance → nearest | UNKNOWN   │
│      │ gloss token (hold-to-confirm ring, with cooldown)                         │
│      ▼                                                                           │
│  Gloss sequence  ["ME","NEED","METFORMIN"]                                       │
│      │                                                                           │
│      ├── seen before ─► sentenceMemory.json     previously chosen sentences      │
│      └── new ────────► llama.rn + Qwen2.5-3B    2–3 candidate sentences          │
│                              │                                                   │
│                              ▼                                                   │
│                   ┌────────────────────────┐                                     │
│                   │  You pick one          │  pick · edit · discard              │
│                   └────────────────────────┘                                     │
│                              │ confirmed                                         │
│                              ├──► remembered for this sign sequence              │
│                              ▼                                                   │
│                   react-native-tts → speaker  ── or ──► LAN relay → peer phone   │
│                                                                                  │
│  Inbound: mic → PCM stream → whisper.rn (ggml-tiny) → transcript on screen       │
│           (mic pauses while the phone is speaking, to avoid self-transcription)  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

| Path | Contents |
|---|---|
| `app/` | The React Native app |
| `app/src/recognition/` | `gestureRecognizer`, `captureQuality`, `duplicateDetection`, `userGestureStore`, `useAllGestureTemplates`, `useLiveHandGestures` |
| `app/src/llm/` | `LlmProvider`, `LocalLlmProvider`, `useLocalLlm`, `prompts`, `sentenceMemory` |
| `app/src/speech/` | `LocalSpeaker` (TTS), `LocalVoiceRecorder`, `LocalWhisperTranscriber` |
| `app/src/relay/` | `AslRelaySender`, `AslRelayReceiver`, zeroconf hooks |
| `app/src/screens/` | Home · Call · Receive · Memory · AddSign |
| `app/src/components/`, `app/src/theme/` | UI primitives and design tokens |
| `app/android/app/src/main/java/com/mudraapp/recognition/` | `HandLandmarkerHelper`, `HandLandmarksFrameProcessorPlugin` (native MediaPipe bridge) |
| `memory-layer/` | Kotlin/Android translation memory — standalone Gradle build, Room + ADPF |
| `memory-layer-rn/` | TypeScript port of the same engine, zero runtime dependencies |
| `sign-embeddings/` | TFLite-embedding custom-sign path (built, tested, not wired in) |
| `llm-testbed/` | Standalone RN app for benchmarking GGUF models on the device |
| `PLAN.md`, `TRAINING.md` | Original plan and classifier-training plan — the path the build moved away from |
| `FEATURE.md`, `asl-call-relay.md` | Recognition writeup; call-audio architecture notes |

---

## Recognition

**Landmarks.** A VisionCamera frame processor calls the native MediaPipe plugin, which returns 21
`{x, y, z}` points per detected hand.

**Features.** Points are normalised around the wrist — translation- and scale-invariant — then
expanded into a **72-dimension vector** using the same transform the snapshot trainer used, so
bundled and user-recorded templates are directly comparable.

**Matching.** Templates ranked by RMS Euclidean distance; `UNKNOWN` above the configured threshold
(default 0.42). Rejection is a feature: an honest `UNKNOWN` is recoverable, a confident wrong gloss
is not.

**Capture quality gate** (`assessCapture`), before any capture is accepted:

| Check | Rejection |
|---|---|
| Landmark shape valid | `no-hand` — "Hand not detected" |
| Hand-to-camera distance (`handSpread`) | `too-far` — "Move your hand closer" |
| Steadiness across the hold window (`maxJitter`) | `unstable` — "Hold your hand steady" |

Steadiness is judged over a rolling buffer of samples covering the whole hold, not just the frame
the ring completes on.

**Hold-to-confirm.** `SignGuideCircle` fills over the hold duration; a cooldown forces the hand to
visibly re-form before the same gloss can fire again. On the call screen a *detection zone*
restricts matching to a hand held at sign height near the guide ring, and consecutive repeats of
the same held sign are de-duplicated.

**User templates.** Recorded signs persist to an app-private JSON file via atomic
temp-file-then-swap writes, with a pub-sub so other mounted screens reload live and a serialised
save queue so concurrent captures cannot corrupt the store. A user label shadows a bundled one of
the same name; saving over an existing label asks for confirmation first.

**Demo limits:** single hand, static poses, one sign per hold. Two-handed and motion-based signs
are not implemented.

---

## Autocomplete

Two stores, different jobs.

### Sentence memory — `app/src/llm/sentenceMemory.ts`

What the shipped app uses. Records **which candidate the user picked** for each exact gloss
sequence and resurfaces it ahead of fresh generation. A sequence can legitimately mean different
things on different occasions, so picking a different sentence later *adds* an option rather than
replacing the earlier one. Once a sequence has enough remembered options, the LLM is not called for
it again. Stored as app-private JSON with the same atomic-write approach as the gesture store; the
**Memory** screen reviews and forgets entries.

### Translation memory — `memory-layer/`, `memory-layer-rn/`

A standalone engine implemented twice — Kotlin (Room/SQLite, ADPF hints) and TypeScript (any
SQLite binding via a six-line adapter) — for corpora larger than a JSON file should hold.

**Exact path — O(1).** Kotlin hashes the token characters in place (FNV-1a, no joined key string,
zero allocation) into an open-addressing table; collisions resolve by *verifying tokens*, never by
trusting the hash. The TS port uses a `Map` on the joined key, the idiomatic O(1) in JS.

**Fuzzy path — two stages**, so order-aware scoring only ever sees a few dozen records:

1. An inverted-index walk accumulates IDF-weighted overlap into a reused, generation-stamped
   `Float64Array`. Tokens appearing in >60 % of memories are skipped unless the query has nothing
   else. Unknown glosses get single-edit repair (`METFORMIM → METFORMIN`) against a length-bucketed
   vocabulary.
2. The top 48 candidates are scored
   `0.45·weightedJaccard + 0.25·queryCoverage + 0.30·lcsRatio` — the LCS term makes sign *order*
   count. Ties break by use count, then recency, then id.

**Measured** (CI container, not the phone):

| | Kotlin | TypeScript |
|---|---|---|
| Exact lookup | ~1 µs | ~2.1 µs |
| 1 k → 50 k memories | flat | flat |
| Tests | 20 JVM + 3 Robolectric | 23 Jest (SQLite against a real engine) |

**Failure behaviour.** Ids are assigned in RAM, so a confirmed phrase is queryable before SQLite
has heard of it; persistence is write-behind. A corrupt or full database flips the layer to
memory-only, counts the failure and notifies the app. Warm-up skips corrupt rows instead of failing
start-up. Malformed recogniser output is a returned value, never an exception. Capacity is bounded
with least-used/least-recent eviction; pinned entries are never evicted.

---

## LLM layer

`LlmProvider` is a one-method interface (`complete(system, user)`); `LocalLlmProvider` implements
it over `llama.rn` with `n_gpu_layers: 99`. The app ships local-only — no cloud provider is wired
in and there is no API key anywhere in the live path.

- Prompts request **multiple candidates**, because a gloss sequence is genuinely ambiguous
  (`ARRIVED HOME` differs between a customer and a driver). The user chooses; the app does not
  guess on their behalf.
- Call conversation history is fed back as context so replies fit what was just said.
- Output is a candidate, never an action — it reaches speech only through the confirmation step,
  which is also the guard against a hallucinated or misrecognised sentence.

---

## Audio and the relay

Android exposes **no public API for injecting audio into a cellular call**, which constrains the
whole output stage. Two routes are implemented:

| Route | How | Trade-off |
|---|---|---|
| **Acoustic coupling** | Speakerphone; the call's own open mic picks up the TTS output | Works on any carrier call with no infrastructure; lower quality, and handset echo cancellation can suppress it |
| **LAN relay** | zeroconf/mDNS (`aslrelay`) + TCP on port 12345; the sender transmits text, the receiver speaks it and streams a transcript back | Clean audio, no cloud telephony; both phones must share a network |

Inbound speech goes mic → `@fugood/react-native-audio-pcm-stream` → `whisper.rn` → on-screen
transcript. The receive mic pauses while the phone is speaking, so the app does not transcribe its
own voice back into the conversation.

---

## Models

| Model | Artifact | Runtime | Role |
|---|---|---|---|
| MediaPipe Hand Landmarker | `hand_landmarker.task` (committed) | MediaPipe Tasks, native | 21 3-D landmarks per frame — the only vision model in the live path |
| Qwen2.5-3B-Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf`, ~2 GB (you supply) | `llama.rn` / llama.cpp | Gloss sequence → candidate sentences; contextual replies |
| Whisper tiny | `ggml-tiny.bin`, ~75 MB (committed) | `whisper.rn` / whisper.cpp | On-device transcription of the other party |
| Android TTS | system voices | `react-native-tts` | Speaks the confirmed sentence |
| Gesture templates | `custom_gestures.json` + user file | pure TypeScript — **no model** | 21 landmarks + 72 features per sign; bundled set plus user recordings |
| *(not wired in)* | `sign_embed.tflite` | TFLite | Embedding extractor for the cosine-similarity custom-sign path |

---

## On-device notes — Snapdragon 8 Elite Gen 5

- **Model size.** Q4_K_M at ~2 GB fits comfortably in the iQOO 15's RAM, which is why the demo uses
  a 3B rather than the 1.5B originally planned. Inference is configured with `n_gpu_layers: 99`,
  and the Android build ships llama.cpp's Hexagon HTP libraries
  (`android/app/src/main/assets/ggml-hexagon/`) — whether the HTP backend engages at runtime
  depends on the `llama.rn` build and the device, so treat NPU offload as available-but-unverified
  rather than a measured win.
- **ADPF hints.** The memory layer's `AdpfPerformanceGovernor` opens a `PerformanceHintManager`
  session per lookup thread (API 31+), declaring a ~2 ms target and reporting actual duration. On a
  big.LITTLE part a microsecond-scale burst otherwise completes on an efficiency core *before* a
  reactive governor reacts. Degrades to a no-op where unsupported.
- **Thread placement.** The memory layer's write-behind thread runs at
  `THREAD_PRIORITY_BACKGROUND` so SQLite never contends with the camera pipeline for prime cores.
  Room opens WAL with `synchronous = NORMAL` on its own executor.
- **Thermals set the budget.** Continuous camera plus continuous LLM is what heats the phone, so
  the design optimises for *fewer* LLM invocations rather than faster ones.
- OriginOS performance-mode toggles are not app-accessible APIs; ADPF is the supported route to the
  same scheduler.

---

## Testing

```bash
cd app              && npm test      # recognition, LLM helpers, screen smoke tests
cd memory-layer     && ./gradlew :core:test
cd memory-layer-rn  && npm test
cd sign-embeddings  && npm test
```

| Suite | Covers |
|---|---|
| `app` | Capture gating, duplicate/collision logic, merged templates, gesture store, LLM helpers, screens |
| `memory-layer` | Retrieval, near-miss ranking, malformed input, restart, corrupt rows, failing database, capacity, concurrency, flat-scaling lookups |
| `memory-layer-rn` | The same behaviours, plus the SQLite store against a real engine |
| `sign-embeddings` | Matching, priority and collision logic on synthetic vectors |

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Model not found" on launch | The `.gguf` is missing. The screen prints the exact `adb push` command; the filename must match `MODEL_FILENAME` in `src/llm/useLocalLlm.ts`. |
| Camera preview is black | Camera permission denied, or the app is on an emulator. Grant it in system settings, and use a physical device. |
| "Hand not detected" / "Move your hand closer" / "Hold your hand steady" | The capture gate working as intended — improve lighting, fill more of the frame, hold still for the whole ring. |
| A sign is never recognised | Its template is too close to another, or the pose drifts from the recording. Re-record it under the same label (the app asks before overwriting). |
| Release build OOMs in `compressReleaseAssets` | AGP choking on the multi-GB model asset — raise the Gradle daemon heap, or use `scripts/build-release-with-llm.ps1`, which does it temporarily. |
| Windows release build fails with long-path errors | VisionCamera's release CMake build hits `MAX_PATH`; the same script builds through a short directory junction. |
| Two phones don't find each other | mDNS blocked (guest Wi-Fi, client isolation) — use a phone hotspot. Both must be on the same network; the session uses TCP port 12345. |
| The receiver transcribes its own TTS | Should not happen — the mic pauses while speaking. If it does, the speaker is likely routed to an external device that re-enters the mic path. |
| Metro cache weirdness after dependency changes | `npm start -- --reset-cache` |

---

## Status and limitations

Completed at the end of the hackathon. Demo-quality, and specific about which parts are which.

| Area | State |
|---|---|
| Live hand tracking, template matching, runtime sign capture | Working on device |
| On-device candidate generation (Qwen2.5-3B) | Working |
| Sentence memory | Working |
| Translation memory, both ports | Working, 46 tests, measured |
| LAN relay, transcription, TTS | Working, real connection state |
| Acoustic coupling into a carrier call | Works, subject to handset echo cancellation |
| Vocabulary | Small bundled set + user recordings, by design |
| Two-handed / motion signs | Not implemented |
| `sign-embeddings` path | Built and tested, not wired in |
| Cloud telephony | Deliberate non-goal |

**Known limitations.** Static single-hand poses only; recognition quality depends on lighting and
framing; the capture-gate thresholds are hand-tuned rather than learned; the relay assumes a shared
network; all timing figures are engineering benchmarks, and none of the ergonomics have been
evaluated with signers yet.

**Next, in order.** Ergonomic evaluation of the hold-and-confirm loop with signers; motion and
two-handed signs; measured per-sign accuracy on the bundled set; wiring `sign-embeddings` if
template matching saturates.

---

## Credits

MediaPipe Tasks · llama.cpp via `llama.rn` · whisper.cpp via `whisper.rn` · Qwen2.5 (Alibaba) ·
React Native · VisionCamera · Room/SQLite.

Built at **iQOO City Battles, Chennai**, on an iQOO 15.
