# Progress and next steps

Last updated: 2026-10-13. Repo history starts at the scaffold commit; earlier work lived on the `feature/sign-language-project` branch of ChatBox-companion-robot and was moved here with its commits.

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

Total: 45 tests, standard library only. Run from the repo root: `python3 -m unittest discover -s interpreter/tests -t . -v`.

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

## Next steps

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
