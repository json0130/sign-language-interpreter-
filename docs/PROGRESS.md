# Progress and next steps

Last updated: 2026-10-16. Repo history starts at the scaffold commit; earlier work lived on the `feature/sign-language-project` branch of ChatBox-companion-robot and was moved here with its commits.

## What this project is

Pepper becomes an interpreter between a Turi Māori / NZSL signer and a non-signer, in both directions. This repo holds the decision layer: how Pepper speaks and shows emotion (Task 1), and what the sign generator is told (Task 2). Sign recognition and sign generation are other people's modules. Pepper uses voice and eye colour only. Language is NZ English.

- **Task 1:** signer to non-signer (`sign_to_speech`).
- **Task 2:** non-signer to signer (`speech_to_sign`).
- **Rule:** interpret, never reply. The emotion shown is the speaker's, never the robot's persona. Pepper's own voice is only for clarification.

## Done

| Step | What | Tests |
|---|---|---|
| 0 | Contracts v0.1 (`contracts.py`): `InterpretationFrame` in, `SpeechPlan` (Task 1 out), `SignPlan` (Task 2 out). Every object validates itself on construction. `SignPlan` is a proposal for the sign-generation owner. | 14 |
| 1 | 10 hand-written mock frames (5 per direction), mock recogniser, `MockPepper`, `StubSignGenerator`, pass-through baselines, `Pipeline`, `python3 -m interpreter.run_mock`. | +6 |
| 2a | Task 1 rule-based planner (`task1.py`): clarify when `input_confidence < 0.5`, te reo word choice via a `ContextProvider`, prosody from arousal/valence scaled by expressiveness x emotion confidence, eye-colour palette, pronunciation lexicon. Realistic emotion confidences added to the mock frames. | +12 |
| 2b-i | TTS adapters (`interpreter/tts/`): NAOqi tagged string, Amazon Polly SSML (neural), Azure SSML. String builders only, no network or keys. Optional `ipa` field on `LexiconEntry`. Preview: `python3 -m interpreter.run_tts_preview`. | +13 |

| 2b-ii | Pepper test bench (`interpreter/robot/`): allowlisted speech and LED transport, read-only probe, listening-test pack. Paused; not run on the robot yet. | +12 |
| L1 | Local speech output with Piper (`interpreter/local_tts/`): rate and pitch from `length_scale` plus a sample-rate relabel, pauses, volume, `run_local`. | +12 |
| C2 | Language-aware contracts (schema 0.2: `text`, `source_lang`, `target_lang`, `Concept.glosses`), `concepts.py`, voice registry (en-NZ, en-US; zh-CN reserved). | +23 |
| L2 | Stronger per-emotion prosody profiles (`prosody_profiles.py`), expressiveness dial, `--sweep`. Piper alone was judged too subtle. | +13 |
| B2 | TTS emotion bake-off (`interpreter/bakeoff/`, report in `REPORT.md`): by ear IndexTTS-2.5 best (too exaggerated at full strength), CosyVoice second, Kokoro/Parler/Chatterbox out. Engine choice and integration parked. | |
| K1 | Stdlib knowledge graph (`interpreter/kg/`) and `KGContext`: does the listener know this kupu? | +10 |
| P1 | End-to-end pipeline (`run_pipeline`): emotion backup rules, word choice by listener ("whānau" to "family" unless the listener knows it), Piper TTS. Fixed `text_to_speech` routing. | +22 |
| T1 | Test sentences in one CSV (`mocks/testset.csv`, 16 rows) and `run_pipeline --testset/--text ... --play` with a summary table; six more common kupu in `concepts.py` (kai, mahi, tamariki, whare, kōrero, koha; needs review). | +18 |
| F1 | Emotion flow (`emotion_flow.py`), standalone: per-clause emotion from markers, rules or the sentence emotion, inertia, per-speaker carry-over, blended delivery, single-vs-flow A/B rendering. | |
| T2/F2/V1 | 26 test sentences (pure English or English with KG-replaceable kupu); emotion flow wired into Task 1 and the renderer (schema 0.3: `emotion_flow` spans on the frame, `segments` on the plan); Kokoro as the default local voice (`--tts piper` still available); Māori pronunciation respellings dropped from the speech path. | +9 |
| S1 | `run_pipeline --step` (one sentence at a time: next, previous, replay, jump, quit) and `--score` (the seven-check marking scheme after each sentence, saved to CSV, with a pass-rate report). | +5 |
| I1 | IndexTTS-2.5 wired in as `--tts indextts`: per-clause emotion vector and strength from the emotion flow, served from its own venv by a local HTTP service that the pipeline starts on demand; quieter library warnings in the trace. Not yet listened to through the pipeline. | +10 |
| I2 | Jay's scoring (2026-10-19, `listening_results/`): IndexTTS 19/26 sentences passed every check (words 26/26, emotion 9/10, flow 16/16, natural 24/26); Kokoro failed on emotion. IndexTTS made the default; clause silence trimmed before joining (about 1.2 s less dead air per flow sentence); service moved to port 8790 (8765 clashed with the ChatBox graph viewer); keep-alive command. | +2 |
| I3 | Draft CMU phoneme hints for the nine kupu sent to IndexTTS for te reo listeners (to be checked by a te reo speaker; `--no-kupu-hints` to compare); the service chunks text itself so hints are never split (IndexTTS's low-VRAM splitter broke them); flows softened to 0.8 strength. | +6 |
| I4 | Kupu hints rebuilt from published pronunciations (Wiktionary Māori IPA, Victoria University of Wellington guide): short a is AH, long ā is AA (the first draft used AA everywhere, which is why whare sounded wrong). Audition tool `python -m interpreter.local_tts.indextts_phonemes <kupu>` / `--changed` plays numbered candidates. | +3 |
| E1 | Grammar-vs-emotion guard: in a plain question, face readings that eyebrow grammar can produce (surprised for yes/no; angry, confused for wh) are set aside as neutral; words can still add emotion. Five question test sentences (g01–g05). Kupu pronunciation parked (needs a te reo voice, e.g. Te Hiku Media's). | +9 |
| R1 | IndexTTS on the AI server: read-only preflight (`local_tts/server_check.sh`), installer pinned to the tested commit (`server_install.sh`), `--indextts-url` / `INDEXTTS_URL` to use a running service through an SSH tunnel (never starts a local copy); readable out-of-memory message (Ollama was holding 4.6 GB of the 8 GB card). Not yet run on the server. | +6 |
| R2 | Speech requests go to the AI server over the Library repo's orchestrator protocol (`interpreter/ai_server/`, default engine `--tts aiserver`; `AI_HOST` / `AI_PORT`); a probe checks the server really is IndexTTS before emotion settings and kupu hints are trusted; our IndexTTS service answers the protocol on port 7899 (tested with the Library repo's own unmodified `tts.py` client). Local engines stay as `--tts indextts / kokoro / piper`. Awaiting the install on the server. | +12 |
| R3 | Live test of the AI server's IndexTTS orchestrator (port 8791), scored by Jay: words and naturalness perfect, but "no emotion felt". Measured: the orchestrator ignores the emotion fields (accepts invalid ones, echoes parameters; sad = 0.97x neutral). Probe rewritten to test behaviour (invalid emo_vector must be refused) instead of trusting the echoed `engine`; `python3 -m interpreter.ai_server.check` conformance check (our service passes, 8791 fails 4 of 5); `ai_server/SERVER_TTS_SPEC.md` for the server's Codex. Server fixed the same day: 8791 passes all five checks (sad = 1.35x neutral), pipeline run clean, about 5 s per clause; Jay to re-score. | +5 |
| R4 | After Jay scored the working AI server (22/31; emotion present, now "a bit too much" in places): a clause that follows a different emotion is softened by the size of the jump (excited then sad = half strength) instead of a flat 0.8; strengths retuned (sad .6, surprised .65, apologetic .65, anxious .8); pauses after ? . ! are longer (450/300 ms); the kupu audition tool uses the server and offers more alternatives for koha and whare. Jay's audition picks: koha `K OW1 . HH AH0`, whare `F AH1 . D EH0`, tamariki the D version. | +6 |

Total: 235 tests, standard library only except the optional Piper venv and the bake-off tooling. Run from the repo root with the venv Python: `interpreter/.venv/bin/python -m unittest discover -s interpreter/tests -t . -v`.

### What the Task 1 output looks like

Against the no-context baseline (all multipliers 1.0, white eyes), `plan_speech` gives:

- f-001 (excited): "My whānau is coming to the hui tomorrow.", rate 1.10, orange pulse, "whānau" stressed.
- f-003 (sad): slower and quieter, dim blue.
- f-004 (angry): faster and louder, red, "nobody" stressed.
- f-005 (input confidence .35): clarify mode, robot voice, amber pulse.

### Findings so far

- Emotional delivery is mild: with expressiveness 0.8 and the mock confidences, the largest prosody change is about 11%. Probably right for an interpreter, but it is the first thing to tune against listener feedback.
- Angry pitch barely moves (f-004 is 1.03): the formula adds 0.15 x arousal and 0.10 x valence, and negative valence cancels most of the arousal term.
- "whānau is": word swapping does no grammar rewriting. Known limitation.
- Only single-word emphasis targets are stressed, because the contract cannot stress a phrase.

## Placeholders and unverified items

Nothing below has been checked on a real robot or with a real speaker. Do not treat any of it as validated.

- **Māori content:** concept tags (`ck:whanau`, `ck:hui`), settings, and the pronunciation respellings in `lexicon.py` ("FAH-no", "HOO-ee", "kee-ah OR-ah") need review by Māori language advisers or speakers. The IPA table is empty on purpose; values must come from a te reo speaker or adviser.
- **Eye colour:** the palette is a placeholder (colour-emotion associations vary between people and cultures). The amber pulse as a visual "please repeat" cue must be validated with Deaf users, after consultation.
- **NAOqi:** `\emph` tag semantics, how the respellings are read, whether Pepper has an NZ English voice, and the base volume (80).
- **Polly / Azure:** neural prosody limits, Azure en-NZ voice name and style support.
- **Mock data:** all frames are fictional, written by hand.

## Next steps (as of 2026-10-15)

The numbered list below is from 2026-10-13 and partly done (steps 1 and 4 are overtaken by 2b-ii and K1/P1). Current priorities: a directed person-to-person relationship edge in the KG (relation type, derived inverse, familiarity, register hint); Task 2 (speech to sign); an emotion TTS integrated behind the speech sink (IndexTTS-2.5 at about 0.5 to 0.7 strength per emotion; its happy and surprised mappings need work).

1. **Step 2b-ii: Pepper test bench.** Allowlisted transport (speech and LEDs only), `PepperOutput`, a read-only probe, a listening-test pack built from the mock frames, and a CSV-writing listening test. First action on the robot is the read-only probe: it answers whether an NZ voice exists, whether `FaceLeds` is available, and whether Autonomous Life is on (it can override eye colours).
2. **Run the listening test** with hearing NZ English listeners, then choose the TTS backend (NAOqi vs Polly vs Azure) from the results, and tune expressiveness and the prosody formulas.
3. **Task 2: speech to sign planner.** Replace `baseline_task2`: resolve idioms ("push the meeting back" = postpone), attach Māori concept signs (`prefer_sign`), set register, emotion intensity and non-manual hints, and lower confidence or flag clarification on poor transcripts. Agree the `SignPlan` fields with the sign-generation owner first.
4. **Context from the KG:** replace `StaticContext` with a KG-backed `ContextProvider` (the culture layer lives in the ChatBox repo, so decide whether it is called as a service or exported). Task 1 already only talks to the interface.
5. **Wire up the real modules:** the sign recogniser (frames in) and sign generator (plans out), replacing the mocks.
6. **Evaluation with Deaf and Māori communities:** colour cues, word choice, and the clarification behaviour, after consultation and with proper ethics approval where required.

## Open decisions

- TTS backend (needs the listening test).
- `SignPlan` fields (needs the sign-generation owner).
- Whether Autonomous Life should be changed for the colour cues (never changed by this code; your call after the probe).
- How the KG culture layer reaches this repo.
