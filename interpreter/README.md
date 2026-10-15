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
interpreter/.venv/bin/python -m piper.download_voices --download-dir interpreter/models en_GB-alan-medium en_US-lessac-medium
```

Run:

```
interpreter/.venv/bin/python -m interpreter.run_local --flat          # renders interpreter/audio_out/<lang>/<frame_id>.wav (+ -flat.wav)
interpreter/.venv/bin/python -m interpreter.run_local --target-lang en-US --flat   # speak en-US instead of each frame's own target_lang
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

## Languages

Every message is language-tagged (`contracts.LANGS`, schema 0.2). `InterpretationFrame` has `text`, `source_lang` and `target_lang` (all default `en-NZ`); the old `text_en` key still loads and `frame.text_en` is a deprecated read-only alias. `Voice.language` and `Concept.glosses` use the same codes. Schema 0.1 messages are accepted and come out as 0.2.

| Language | Status | Local voice |
|---|---|---|
| en-NZ | Supported | `en_GB-alan-medium` (British stand-in until an NZ voice is chosen) |
| en-US | Supported | `en_US-lessac-medium` |
| zh-CN | Reserved, not implemented (`plan_speech` and `engine_for` raise `NotImplementedError`) | none |

`plan_speech` accepts `sign_to_speech` and `text_to_speech` frames and speaks in `target_lang` (argument, else the frame's). `interpreter/concepts.py` holds the `ck:` concept table (te reo form plus per-language glosses). All glosses are placeholders that need review by a te reo speaker or adviser, and the zh-CN glosses are empty until fluent speakers supply and review them.

## Pipeline (transcript to spoken English)

```
sign transcript -> emotion backup -> KG context -> Task 1 plan -> local TTS
   (frame)         emotion.py        kg/            task1.py       local_tts/
```

Run it from the repo root. Every frame prints numbered stages showing each decision and why (input, emotion, KG, plan, speech):

```
python3 -m interpreter.run_pipeline --demo --dry-run          # the te reo demo, stages only; plain python3 is enough
interpreter/.venv/bin/python -m interpreter.run_pipeline --demo          # same, and renders the WAV (not played)
interpreter/.venv/bin/python -m interpreter.run_pipeline --demo --play   # render and play it
interpreter/.venv/bin/python -m interpreter.run_pipeline                 # all 10 mock frames -> audio_out/pipeline/<lang>/
python3 -m interpreter.run_pipeline --dry-run --only f-003,f-005,f-006   # chosen frames only (a KG-known listener, a clarify, a sign frame)
python3 -m interpreter.run_pipeline --dry-run --quiet                    # one line in, one line out per frame
```

Other flags: `--kg FILE` (a graph saved with `Graph.save`; default is the demo graph), `--target-lang en-NZ|en-US`, `--expressiveness 0..2`. What to check in the trace: stage 2 says whether the recogniser's emotion was kept or replaced by a text rule (and which words matched); stage 3 lists each kupu, whether this listener knows it and why (a `knows_kupu` edge, or a culture tag with a prior), and the word it will say; stage 4 shows the final sentence, the emotion profile and strength, and the resulting rate, pitch and volume; stage 5 shows the audio length and file.

- **Knowledge graph** (`interpreter/kg/`, standard library only): people, cultures and `ck:` concepts, with three edge kinds: `belongs_to_culture` (a tag, never a fact about the person), `culture_prior` and `knows_kupu` (weights 0 to 1). Every edge records its provenance (source, confidence, time), newer writes replace older ones, and the graph saves to JSON (`Graph.save` / `Graph.load`). `kg/demo.py` builds a small demo graph matching the mock frames. A stdlib port of the idea behind the ChatBox V5 graph module; the V5 copy is kept untracked in `kg/_v5_reference/`.
- **Word choice follows the listener** (the frame's `addressee_id`): `KGContext.listener_knows_te_reo` is true when the listener has a `knows_kupu` edge at or above 0.5, or belongs to a culture whose prior for the kupu is at or above 0.5. If so the kupu is spoken ("whānau"); otherwise the plain English word is ("family"), whichever form the transcript used. Multi-word kupu such as "kia ora" are found, ignoring case and macrons. Because the rule is per listener, "Kia ora" becomes "Hello" for a listener the KG does not know.
- **Emotion backup** (`emotion.py`): the recogniser's emotion is kept when its confidence is at least 0.4. Below that, simple English text rules (sorry, thanks, worried, and so on) give a label at confidence 0.5; if no rule matches the frame is unchanged.
- **Speech:** Piper through `local_tts`. Piper conveys emotion only through rate, pitch and volume, which listeners found too subtle, so a TTS engine with real emotion input is still to be chosen (see `bakeoff/REPORT.md`; IndexTTS-2.5 is the shortlisted engine).

Placeholders: the demo graph's culture priors are guesses, and which kupu a community or person knows needs review by Māori language advisers or speakers.

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

Status: Steps K1 (stdlib knowledge graph) and P1 (end-to-end pipeline) added; awaiting sign-off. TTS emotion engine parked (IndexTTS-2.5 shortlisted).
