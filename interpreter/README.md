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

Status: Step 2b-ii (Pepper NAOqi test bench) added; awaiting sign-off. Next: run the probe and listening test, then decide the TTS backend.
