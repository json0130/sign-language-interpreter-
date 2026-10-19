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

## Test sentences, emotion flow and voice

**Test sentences** live in one file, `interpreter/mocks/testset.csv` (31 rows). Every sentence is pure English or English with a few kupu Māori that the KG can replace (kia ora, whānau, hui, kai, mahi, tamariki, whare, kōrero, koha), in five groups: `e` pure English, `k` English with kupu for different listeners, `f` sentences where the signer's emotion changes part way (written as `{emotion}` markers), `r` sentences whose clauses change emotion through their own words, and `g` questions for the grammar-vs-emotion guard. Columns: `id, mix, text, emotion, confidence, listener, concepts, notes` (see `interpreter/testset.py`).

```
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --play                  # every sentence: stages, then spoken
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --only f01,f02,r01 --play
interpreter/.venv/bin/python -m interpreter.run_pipeline --text "{happy}Kia ora, {sad}but my nan could not come." --play
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --only f01 --voice af_heart --play   # try another voice
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --tts piper --play      # the old Piper voice
```

**One at a time, and scoring.** `--step` plays one sentence, then waits: Enter or `n` = next, `p` = previous, `r` = replay, `q` = quit, or type an id (for example `f05`) to jump. `--score` does the same and, after each sentence, asks the marking-scheme questions that apply (answer `y`, `n` or `s` to skip; `r` replays). Answers are saved after every sentence to `interpreter/listening_results/scores-<voice>-<time>.csv` (git-ignored; `--score FILE` picks the file), and the run ends with the pass rate and the yes rate per check.

```
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --step
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --score
interpreter/.venv/bin/python -m interpreter.run_pipeline --testset --score --tts piper     # score the old voice the same way
```

| Check | Yes means | Asked for |
|---|---|---|
| Words | what you heard matches the `says:` line exactly | every sentence |
| Māori words | none for an English listener; kept only where the KG says the listener knows them | every sentence |
| Emotion | with your eyes closed you would name the intended emotion | one-clause sentences |
| Flow | the change at each clause boundary is audible and goes the right way | sentences with an emotion flow |
| Natural | sounds like a real person (not robotic, chipmunky, dragging or rushed) | every sentence |
| Joins | no clicks, odd gaps or volume jumps between clauses | sentences with an emotion flow |
| Not overdone | the emotion is not cartoonish or exaggerated | every sentence |

A sentence passes when every check asked is yes (any no fails it; a skip makes it incomplete).

Each sentence prints its "listen for" note and the five stages, then plays. Stage 4 shows the **emotion flow**: the sentence split into clauses, each with its own emotion (from the recogniser's span, a text rule, or the sentence emotion), valence/arousal and delivery (rate, pitch, volume). Stage 5 shows the voice and what was applied to each clause. A summary table ends the run. `--carry` carries each speaker's emotion from one sentence to the next (off by default, so every test sentence starts fresh). Listeners in the demo graph: `person:aroha`, `person:rawiri` (tagged Māori: te reo kept), `person:staff_01` (knows whānau only), `person:visitor_01`, `person:visitor_02` (no te reo: kupu replaced with English).

**Grammar vs emotion** (`guard_grammar` in `interpreter/emotion.py`): in NZSL, as in many sign languages, the face also does grammar. Raised eyebrows can mark a yes/no question and furrowed eyebrows a wh-question, and a face-reading emotion recogniser can mistake them for "surprised" or "angry"/"confused". When a sentence (or one emotion span of it) is a plain question, those look-alike readings are set aside as neutral with 0.4 of their confidence; the words can still add an emotion through the text rules ("Wow, really?" stays surprised). Stage 2 of the trace says when this happens. The look-alike table needs confirming with Deaf advisers. Test sentences g01–g05 cover it.

**Emotion flow** (`interpreter/emotion_flow.py`; contract schema 0.3: `InterpretationFrame.emotion_flow` spans in, `SpeechPlan.segments` out). Clauses break at `. ! ? ;`, before ", but" / ", so" / ", although" / ", and then", and where a recogniser span starts. Unmarked clauses keep 30% of the previous clause so the voice does not jump. Delivery is a smooth blend of the emotion profiles. The renderer speaks each clause with its own delivery and joins them with a 120 ms gap. To compare one emotion against the flow for the same sentence:

```
interpreter/.venv/bin/python -m interpreter.emotion_flow --testset --only f01 --play         # plays single, then flow
```

**Voice: IndexTTS is the default** (since Jay's scoring on 2026-10-19: 19/26 sentences passed every check, against Kokoro failing on emotion). It is IndexTTS-2.5, the bake-off winner, and it takes the emotion itself: each clause's emotion becomes an IndexTTS emotion vector and strength (`interpreter/local_tts/indextts_emotion.py`: neutral 0.5, sad 0.7, excited 0.5, angry 0.7 from Jay's listening; happy 0.7 and surprised 0.8 raised after "happy sounds neutral, surprised not surprised"; the rest untuned), clauses between emotions get a blend, and our rate/pitch trick is not applied. Each clause's leading and trailing silence is trimmed before the clauses are joined. The voice is cloned from `interpreter/bakeoff/reference/neutral_ref.wav` (Kokoro bf_emma).

IndexTTS needs its own torch build, so it runs as a small HTTP service (`interpreter/local_tts/indextts_server.py`) in the bake-off's IndexTTS venv, on port 8790 (8765 is the ChatBox graph viewer's). The pipeline starts it on first use (about a minute; log in `interpreter/audio_out/indextts_server.log`) and stops it on exit. To skip the wait on every run, keep it running in a second terminal:

```
interpreter/.venv/bin/python -m interpreter.local_tts.indextts_engine     # Ctrl-C to stop
```

It needs about 5.7 GB of GPU memory. The same server can run on another machine (`--host 0.0.0.0`, then set `INDEXTTS_URL` in `local_tts/config.py`); it has no authentication, so only on a trusted network. `--tts kokoro` or `--tts piper` switch back; Kokoro uses `bf_emma` (en-NZ) and `af_heart` (en-US), set in `KOKORO_VOICES`.

**Kupu pronunciation (IndexTTS):** kupu are replaced with English for listeners who do not know them. For listeners who do, IndexTTS now gets a phoneme hint for each kupu (`<whānau|F AA1 . N AW0>`, from `interpreter/local_tts/indextts_phonemes.py`), because its English voice mispronounced them (Jay's scoring: e02, k02, f08). The hints are built from published pronunciations (Wiktionary's Māori IPA and the Victoria University of Wellington te reo guide; the first draft wrongly used the long "car" vowel for every short *a*), but English phonemes only approximate te reo, so they must be checked by ear; audition the candidates for a kupu with `interpreter/.venv/bin/python -m interpreter.local_tts.indextts_phonemes whare` (or `--changed` for every hint that changed); the trace shows the exact text sent ("sent to IndexTTS: ..."). Compare with and without: `--only e02,k02,k05,f08 --step` and the same with `--no-kupu-hints`. The service chunks text itself so a hint is never split (IndexTTS's own low-memory splitter cut hints at their dots and read the stress digits aloud).

*Status of the hints (Jay's audition, 2026-10-19):* tamariki works with D for the tapped r; whare and kia ora sounded wrong in every version (including no hint); mahi and koha unclear. IndexTTS's English voice cannot say these well, so kupu pronunciation is **parked**. The proper fix is a te reo Māori voice, such as Te Hiku Media's te reo TTS (Papa Reo); access goes through Te Hiku Media under their data-sovereignty terms, which need checking with them or advisers.

**Softer flows:** in a sentence whose clauses carry different emotions, every clause's IndexTTS strength is multiplied by 0.8 (`FLOW_STRENGTH` in `indextts_emotion.py`), after Jay found f05 and r05 too exaggerated; single-emotion sentences keep the per-emotion strengths. The trace marks these clauses "softened".

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

Status: grammar-vs-emotion guard added (g01–g05); next: Pepper's body language and emotion-driven delivery.
