# interpreter

The robot is changing from a companion into an interpreter between a Turi Māori / NZSL signer and a non-signer, in both directions. Pepper speaks and shows emotion through voice and eye colour only. Language is NZ English. Sign recognition and sign generation are other people's modules; this branch holds the decision layer that uses the KG, culture layer and PAD to decide how Pepper speaks and what the sign generator is told.

## Directions

- **Task 1:** signer to non-signer
- **Task 2:** non-signer to signer

## Rule

Interpret, never reply; PAD carries the speaker's emotion, not the robot's persona.

## TTS backends

`interpreter/tts/` turns a `SpeechPlan` into a payload string per backend. Strings only: no network, no SDKs, no keys. Preview: `python3 -m interpreter.run_tts_preview --backend all`.

| Backend | Used | Dropped |
|---|---|---|
| NAOqi (`\rspd\ \vct\ \vol\ \pau\ \emph\`) | rate, pitch, volume, pauses, emphasis (optional), lexicon respelling (lowercase, no hyphens) | `ipa`; macrons stripped from all words |
| Polly neural SSML | rate, volume (dB), pauses, lexicon `ipa` | emphasis; pitch (unless `polly_use_pitch`) |
| Azure SSML | rate, pitch, volume, pauses, lexicon `ipa` | emphasis |

Unverified:
- NAOqi `\emph` semantics on Pepper
- how NAOqi reads the respellings
- whether Pepper has an NZ English voice
- Polly neural prosody limits (rate/volume may be ignored)
- Azure en-NZ voice name and style support
- all IPA (the table is empty on purpose; values need a te reo speaker or adviser)


## Local speech output

`interpreter/local_tts/` turns a `SpeechPlan` into a WAV on this machine with [Piper](https://github.com/OHF-Voice/piper1-gpl) (`piper-tts`, the only third-party package; it lives in `interpreter/.venv/` and is imported lazily). The robot and cloud TTS paths are paused, not removed, and `SpeechPlan.eye` is ignored here.

Setup (once, from the repo root):

```
python3 -m venv interpreter/.venv
interpreter/.venv/bin/pip install piper-tts
mkdir -p interpreter/models
interpreter/.venv/bin/python -m piper.download_voices --download-dir interpreter/models en_GB-alan-medium
```

Run:

```
interpreter/.venv/bin/python -m interpreter.run_local --flat          # renders interpreter/audio_out/<frame_id>.wav (+ -flat.wav)
interpreter/.venv/bin/python -m interpreter.run_local --flat --play   # also plays each file
interpreter/.venv/bin/python -m unittest discover -s interpreter/tests -t . -v   # includes the real-Piper test
```

`--flat` renders the same words with default prosody, for A/B listening. Playback uses the first of `afplay`, `paplay`, `aplay`, `ffplay`, `powershell`.

How the prosody is produced (`local_tts/render.py`): Piper has no pitch control, so rate `r` and pitch `p` come from one trick. Synthesise with `length_scale = base * p / r`, then label the WAV with sample rate `round(native * p)`. Playing at `p` times the native rate raises pitch by `p` and speeds tempo by `p`; the longer `length_scale` already stretched the audio by `p`, so net tempo is `r`. Pauses are silence sized against the output rate, so they last the stated milliseconds in real playback. Volume: the whole waveform is normalised to `BASE_PEAK` (0.7), then multiplied by `prosody.volume` (clamped 0.25 to 1.5) and clipped. Pitch is clamped to 0.8 to 1.25. Lexicon respellings are lowercased, with hyphens kept (`RESPELL_STYLE = "hyphen"`) or removed (`"joined"`).

Not supported yet:
- Emphasis (a note is added to the result).
- Any NZ English voice (`en_GB-alan-medium` is a British male voice, used as a stand-in).
- Verified te reo pronunciation: the respellings are unchecked placeholders and need review by Māori language advisers or speakers.
- Exact tempo: Piper's length scale is not perfectly proportional, so measured durations drift from the target rate. Judge by ear.

## Pepper test bench

`interpreter/robot/` plays Task 1 output on a real Pepper (NAOqi) and runs a listening test. Needs the optional `qi` bindings for a real robot (`pip install qi`, or the Aldebaran Python 2.7 SDK); `--dry-run` needs nothing.

```
# 1. Read-only probe (NZ voice? FaceLeds? Autonomous Life state?)
python3 -m interpreter.robot.probe --host PEPPER_IP --out interpreter/listening_results/probe.json

# 2. Rehearse without a robot (prompts and CSV as normal)
python3 -m interpreter.robot.listening_test --dry-run --rater R01

# 3. Real listening test (speaks and changes eye colour)
python3 -m interpreter.robot.listening_test --host PEPPER_IP --rater R01 --seed 1
```

`--no-input` plays every variant with no prompts or CSV (smoke check). Results go to `interpreter/listening_results/` (git-ignored), one CSV row per rated variant.

Safety:
- Speech and LEDs only. Every call is checked against the allowlist in `robot/transport.py` and anything else raises `PermissionError`: no motion, no posture, no Autonomous Life changes, no audio recording, no file access on the robot.
- `emergency_stop()` (stopAll plus resting eye colour) runs at the end of every session and on Ctrl-C.

Autonomous Life: if the probe reports a state other than `disabled`, it may override the eye colours. This code never changes it; whether to change it is your decision.

Privacy and ethics:
- The test is for hearing NZ English listeners, to tune the voice. Identify raters by an ID or pseudonym, not a real name, and get their agreement before recording their ratings.
- The eye-colour cue (and the amber pulse as a "please repeat" signal) must be evaluated with Deaf users separately, and only after consultation and any ethics approval required. Do not use this test for that.

Progress and next steps: [docs/PROGRESS.md](../docs/PROGRESS.md)

Status: Step L1 (local Piper speech output) added; awaiting listening check. Robot and cloud TTS paused.
