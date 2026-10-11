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


Status: Step 2b-i (TTS adapters) added; awaiting sign-off. Next: Pepper NAOqi test bench (2b-ii), then Task 2.
