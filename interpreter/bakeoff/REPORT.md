# TTS emotion bake-off (Step B2)

> **Update (2026-10-15):** Jay's listening verdict is in section 9. The Kokoro, Parler and Chatterbox clips described below were deleted at Jay's request (they are git-ignored and can be regenerated with `run.py`), so the paths in sections 5 and 6 for those three engines no longer exist. The CosyVoice and IndexTTS clips remain; see section 9 for the second round.

Throwaway evaluation. Nothing outside `interpreter/bakeoff/` was changed, and no audio was played by the tooling; the listening is yours. Date run: 2026-10-14.

**Why:** Piper plus our sample-rate pitch shift cannot convey emotion (you heard none in the sweep). This compares open TTS engines that take an emotion or style input.

**Read this first.** The only automatic measure is a speech-emotion recogniser (SER), which is a rough proxy, not ground truth: it was not trained on TTS, has no "excited" class, reports scores near 1.0 even when wrong, and with n=12 per engine one clip is 8 points. Treat the numbers as a hint about where to listen first.

## 1. Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060, 8 GB (about 7 GB free; the desktop was using about 1 GB), compute capability 12.0 (Blackwell, sm_120) |
| Driver / CUDA | 580.178.04 / 13.0 |
| CPU / RAM / disk | Intel Core Ultra 7 265, 20 threads / 30 GB (about 13 GB free) / 630 GB free |
| Consequence | Every venv needs torch >= 2.7 with sm_120 kernels. The system torch (2.2.2+cu121) cannot run on this GPU. Each venv's torch was checked with a CUDA matmul. |

## 2. Candidates (verified live on 2026-10-14)

Sources: each project's GitHub README and Hugging Face model card and file listing (via the HF API); PyPI metadata for chatterbox-tts; and, for IndexTTS only, the full LICENSE text in the cloned repo. Where only a model card was read, the code licence is the card's statement and was not checked separately.

| Engine | Version used | Code licence | Weights licence | Commercial / research | Weights on disk | Languages (see 5) |
|---|---|---|---|---|---|---|
| Kokoro-82M | v1.0 (Jan 2025) | Apache-2.0 (model card; pip package licence not checked) | Apache-2.0 | both allowed | 0.33 GB | |
| Parler-TTS Mini v1.1 | Oct 2024 | Apache-2.0 | Apache-2.0 (HF card) | both allowed; training-data licences not checked | 3.75 GB | |
| Chatterbox Multilingual V3 | chatterbox-tts 0.1.7 | MIT | MIT | both allowed; outputs carry a Perth watermark | about 3 GB used | |
| Fun-CosyVoice3-0.5B-2512 | Dec 2025 weights | Apache-2.0 | Apache-2.0 (HF card) | both allowed. The README adds a disclaimer that the demo content is "for academic purposes only" | 9.75 GB repo (9.1 GB downloaded) | |
| IndexTTS-2.5 | released 2026-08-10 | bilibili Model Use License Agreement | same agreement | use is permitted without a separate licence unless you or an affiliate had >100M monthly users or >RMB 1bn annual revenue. Also: outputs may not be used to improve other AI models except non-commercial ones, high-risk deployments are prohibited, governed by PRC law, Chinese text prevails | 11 GB downloaded | |
| emotion2vec_plus_large (the SER, not a TTS) | | FunASR code licence not checked | FunASR MODEL_LICENSE (Alibaba) | commercial and research allowed with attribution | 1.95 GB | |

IndexTTS-2.5 is the extra candidate I added: it has the most explicit emotion control I found (8-value emotion vector, emotion reference audio, or text-derived emotion), and a stable English and Mandarin story.

**Not run (would not fit the 8 GB GPU, as the brief says to skip):**
- **Orpheus TTS (canopylabs/orpheus-3b-0.1-ft):** Apache-2.0 weights on a Llama 3.2 base. The repo's safetensors are 15.1 GB (fp32); even in bf16 a 3B model is about 6.6 GB before the SNAC decoder and KV cache, against about 7 GB free. It is also gated on Hugging Face. Its emotion control is inline tags (`<laugh>`, `<sigh>`, ...), not our four emotions.
- **Fish Audio S2 Pro (fishaudio/s2-pro):** 4.6B parameters, 11 GB of weights, about 9.9 GB VRAM at FP16. Licence is the Fish Audio Research License (non-commercial; commercial needs a paid licence). Quantised third-party forks exist; not tried.

## 3. Install results

Separate venv per engine under `bakeoff/venvs/`; exact commands in `setup.sh`.

| Engine | Install ok | What was needed |
|---|---|---|
| Kokoro | yes | Python 3.11. misaki's auto-install of the spaCy model did not land in the uv venv; installed the wheel by hand. |
| Parler | yes | Python 3.11, from git; torch overridden to a Blackwell build (pulled torch 2.14.1+cu130). |
| Chatterbox | yes | Python 3.11. The package pins `torch==2.6.0` (no sm_120 kernels), so it was overridden. First run failed with `TypeError: 'NoneType' object is not callable` (perth's watermarker silently failed to import because `pkg_resources` was missing); fixed with `setuptools<81`. |
| CosyVoice 3 | yes | Python 3.10. The repo's `requirements.txt` pins torch 2.3.1, deepspeed and tensorrt, so a trimmed inference-only list was used (`cosyvoice_requirements.txt`). Missing at first run: `pyarrow`, `setuptools<81`, `torchcodec`. |
| IndexTTS | yes | `uv sync --all-extras` (the README command) **failed**: the optional `accel` extra builds flash-attn/deepspeed and needs `CUDA_HOME`. Plain `uv sync` works (Python 3.11, torch 2.8.0+cu128). A harmless "Ninja is required" message appears when its optional CUDA kernel is not built. |

## 4. Method

Three sentences x four emotions (neutral, sad, excited, angry), identical wording across emotions, one WAV each:
- s1 "My whānau are coming to the hui tomorrow."
- s2 "I'm sorry, the meeting has been moved to next week."
- s3 "The results are in."

- **Emotion mapping:** one small dict per engine at the top of `engines/adapter_<name>.py` (exaggeration values, instruction strings, emotion vectors or voice descriptions). These are untuned starting guesses. Where an engine has no slot for an emotion I used the nearest one: CosyVoice "excited" reuses its "very happy" instruction, and IndexTTS "excited" is happy plus surprised.
- **Loudness:** every clip is scaled to the same speech-active RMS (0.08, peak-limited at 0.95), originals kept in `out/<engine>/raw/`.
- **Speaker:** Kokoro uses `bf_emma`; Parler uses a named speaker ("Jon") in every description. For the three engines that clone from a reference clip (Chatterbox, CosyVoice, IndexTTS) I used one shared neutral reference, Kokoro's s2 neutral clip. The repos' own sample prompts carry emotion of their own: the SER labels IndexTTS's `voice_01.wav` angry (score 1.0) and CosyVoice's prompt happy, and that leaked into the output in the first pass. The first pass (repo sample voices) is reported below for comparison.
- **RTF** is synthesis time divided by audio time on the GPU, after an untimed warm-up call. For the cloning engines it includes encoding the reference each call, except where the engine caches it.
- **SER:** emotion2vec_plus_large via funasr. A clip counts as a match if the predicted label is the intended one; for "excited" (no such class) happy or surprised counts. The same matching is used for every engine.

## 5. Results

| Engine | Licence (code / weights) | GPU needed | Install ok | Median RTF (max) s1 / s2 / s3 | Languages (verified how) | Emotion control type | SER match (of 12) | Notes |
|---|---|---|---|---|---|---|---|---|
| Kokoro-82M | Apache-2.0 / Apache-2.0 | no (82M; run on GPU here) | yes | 0.01 / 0.01 / 0.02 | en-GB rendered here. The installed code lists en-US, en-GB, es, fr, hi, it, pt-BR, ja and Mandarin (needs `misaki[zh]`, not installed, not tested) | none (baseline) | 3/12 | By construction identical for all four emotions; all four read as neutral. Peak VRAM 0.67 GB. |
| Parler-TTS Mini v1.1 | Apache-2.0 / Apache-2.0 | practical yes | yes | 5.5 (13.9) / 0.67 (0.68) / 0.72 (18.5) | English only (model card: trained on English audio) | free-text voice description | 4/12 | Erratic speed: most calls take 1 to 2 s, but 4 of 14 took about 29 s each (cause not investigated). Almost everything reads as "sad". Peak VRAM 4.67 GB. |
| Chatterbox Multilingual V3 | MIT / MIT | practical yes | yes (2 fixes) | 0.43 / 0.41 / 0.48 | 23 languages in the installed `SUPPORTED_LANGUAGES`. English rendered. Mandarin smoke render ok; Whisper transcript has the right words (in Traditional script) | one intensity knob (`exaggeration`, `cfg_weight`), no emotion label | 3/12 | Delivery changes in pacing (angry s2 4.02 s vs neutral 3.38 s) but the SER hears neutral almost everywhere. Peak VRAM 3.62 GB. |
| Fun-CosyVoice3-0.5B | Apache-2.0 / Apache-2.0 | practical yes | yes (3 fixes) | 0.59 / 0.54 / 0.89 | README: zh, en, ja, ko, de, es, fr, it, ru plus Chinese dialects. English rendered. Mandarin smoke render: Whisper heard "我的家人明天打开会" (one character off) | instruction text (`inference_instruct2`) | 4/12 | Tested emotion instructions are Chinese phrases (very happy / sad / angry); with the neutral reference it stays mostly neutral. In the first pass (its own Mandarin prompt) it scored 6/12, but that prompt reads "happy" to the SER. Peak VRAM 3.62 GB. |
| IndexTTS-2.5 | bilibili licence | practical yes | yes (after dropping extras) | 0.50 / 0.53 / 0.56 | README: zh, en, ja, es, ar. English rendered. Mandarin smoke render: Whisper transcript matches exactly | 8-value emotion vector, emotion audio reference, or text-derived emotion | 7/12 | Most responsive: angry 3/3, excited 2/3 happy, sad clearly slower (s2 6.03 s vs neutral 4.23 s). But its neutral reads as angry 2/3 times, so the proxy may be hearing its voice character. Peak VRAM 5.67 GB (leaves about 1.3 GB spare on this 8 GB card). |

"GPU needed": every engine was run on the GPU only. I did not time CPU runs.

### SER confusion (intended row, predicted column counts, over the 3 sentences; shared neutral reference)

| Engine | neutral | sad | excited | angry |
|---|---|---|---|---|
| Kokoro | neutral 3 | neutral 3 | neutral 3 | neutral 3 |
| Parler | sad 2, neutral 1 | sad 3 | other 1, neutral 1, sad 1 | sad 3 |
| Chatterbox | neutral 3 | neutral 2, unknown 1 | neutral 3 | neutral 3 |
| CosyVoice | neutral 3 | neutral 3 | neutral 3 | angry 1, neutral 2 |
| IndexTTS | neutral 1, angry 2 | sad 1, angry 1, fearful 1 | happy 2, sad 1 | angry 3 |

First pass (repo sample voices, before the shared reference): Kokoro 3/12, Parler 4/12, Chatterbox 2/12 (its built-in voice), CosyVoice 6/12, IndexTTS 5/12 (its neutral then used the emotion of its own sample clip). In that pass the cloning engines' output largely inherited the emotion of the reference clip.

### Mandarin check (zh-CN is reserved; this only tests the engines' claims)

Text "我的家人明天来开会。" rendered by Chatterbox, CosyVoice and IndexTTS, then transcribed with Whisper small. Files: `out/<engine>/zh/zh_check.wav`, results in `out/zh_asr.json`. This confirms the models produce recognisable Mandarin; it says nothing about quality or naturalness.

### te reo spelling (files to listen to; no judgement made)

For each engine, the raw macron spelling (`te_reo__raw.wav`, "whānau/hui") and the lexicon.py respelling as local_tts uses it (`te_reo__respelled.wav`, "My fah-no are coming to the hoo-ee tomorrow."):

| Engine | Raw | Respelled |
|---|---|---|
| Kokoro | `out/kokoro/te_reo__raw.wav` (2.98 s) | `out/kokoro/te_reo__respelled.wav` (3.23 s) |
| Parler | `out/parler/te_reo__raw.wav` (2.83 s) | `out/parler/te_reo__respelled.wav` (3.04 s) |
| Chatterbox | `out/chatterbox/te_reo__raw.wav` (3.42 s) | `out/chatterbox/te_reo__respelled.wav` (3.58 s) |
| CosyVoice | `out/cosyvoice/te_reo__raw.wav` (3.40 s) | `out/cosyvoice/te_reo__respelled.wav` (3.76 s) |
| IndexTTS | `out/indextts/te_reo__raw.wav` (2.87 s) | `out/indextts/te_reo__respelled.wav` (3.07 s) |

All paths below are relative to `interpreter/bakeoff/`. Pronunciation of kupu Māori needs a te reo speaker or adviser, not this report.

## 6. Listening order (sentence 2, "I'm sorry, the meeting has been moved to next week.")

For each engine: neutral, then sad, then excited, then angry. Suggested engine order: Kokoro first as the no-emotion reference.

1. Kokoro: `out/kokoro/s2__neutral.wav`, `s2__sad.wav`, `s2__excited.wav`, `s2__angry.wav`
2. Parler: `out/parler/s2__neutral.wav`, `s2__sad.wav`, `s2__excited.wav`, `s2__angry.wav`
3. Chatterbox: `out/chatterbox/s2__neutral.wav`, `s2__sad.wav`, `s2__excited.wav`, `s2__angry.wav`
4. CosyVoice: `out/cosyvoice/s2__neutral.wav`, `s2__sad.wav`, `s2__excited.wav`, `s2__angry.wav`
5. IndexTTS: `out/indextts/s2__neutral.wav`, `s2__sad.wav`, `s2__excited.wav`, `s2__angry.wav`

Play a file with `aplay interpreter/bakeoff/out/indextts/s2__sad.wav`. The other sentences use the same names (`s1__...`, `s3__...`). Unnormalised originals are in `out/<engine>/raw/`.

## 7. What this suggests (to confirm by ear)

- Kokoro is a fast, small, permissive neutral baseline with no emotion control, as expected.
- Of the engines with a real emotion input, only IndexTTS-2.5 was recognised as the intended emotion by the SER for some of the non-neutral emotions (angry 3/3, excited mostly happy) and slowed noticeably for sad, and it covers English and Mandarin. Its licence is the least standard of the five and needs reading by whoever owns the decision, and it uses the most VRAM (5.7 GB of 8).
- Chatterbox and CosyVoice, given a neutral reference voice, changed pacing but were heard as neutral by the SER. This may mean the mappings need tuning (CosyVoice's tested emotion instructions are Chinese phrases, and I used those unchanged), or that the reference clip dominates. Not resolved here.
- Parler follows its description loosely and is slow and erratic here; English only.
- None of this tests naturalness, how emotion reads for NZ English listeners, or the Mandarin quality, which need listeners.

## 8. Reproduce

`setup.sh` (commands that worked, read before running), then `python3 run.py --device cuda`. Adapters are `engines/adapter_<name>.py`; `common.py` holds the sentences, the neutral-reference path and the timing loop; `ser.py` and `zh_check.py` are the proxy checks. Outputs, venvs, models, cloned repos and the reference clip are git-ignored.

## 9. Listening verdict and round 2 (2026-10-15)

Jay's verdict on round 1 (sentence 2, by ear):

| Engine | Verdict |
|---|---|
| Kokoro | No difference between neutral, sad, excited and angry (expected). |
| Parler | Some change, but each emotion sounds like a different person, and all sound depressed. |
| Chatterbox | Very natural, with a bit more variation than Kokoro. |
| CosyVoice | Good, second best, except that **excited sounded sad**. |
| IndexTTS-2.5 | The best, but **a bit too exaggerated**. |

Decision: the Kokoro, Parler and Chatterbox clips were removed. CosyVoice and IndexTTS are kept as the two options, each tuned once more before choosing (or both kept as options). Jay also noted that VRAM need not rule an engine out, because the TTS can run on the AI server and be requested over the network. That was not evaluated here: the AI server's GPU and the request path are unknown to this report.

### CosyVoice: excited fixed (`tune_cosyvoice.py`)

CosyVoice 3 documents only three trained emotion instructions in its repo (very happy, very sad, very angry), plus loud, soft, slow and fast. Version 1 reused "very happy" for excited. That clip ran 0.52 s *slower* than neutral on sentence 2, which fits "sounded sad". I rendered eight variants of excited on sentences 2 and 3 (`out/ct_<variant>/s2__excited.wav`, `s3__excited.wav`), scored by the SER and by length relative to neutral:

| Variant | s2 length vs neutral | SER (s2 / s3) |
|---|---|---|
| v1_happy (the old mapping) | +0.52 s | neutral / neutral |
| happy, speed 1.15 | +0.06 s | neutral / neutral |
| happy, speed 1.30 | -0.28 s | unknown / neutral |
| happy + "as fast as possible" instruction | 0.00 s | neutral / happy |
| happy + loud | +0.12 s | neutral / neutral |
| Chinese "very excited" | +0.28 s | neutral / happy |
| **Chinese "excited, joyful tone" (new mapping)** | **-0.64 s** | neutral / happy |
| English "very excited and happy voice" | +1.00 s | angry / happy |

The new mapping is the Chinese phrase 请用激动欢快的语气说一句话 ("speak in an excited, joyful tone"). It is not one of the repo's tested instructions, so it relies on the model generalising. Full 12-clip set re-rendered as `out/cosyvoice/` (the old one is `out/cosyvoice_v1/`); the other three emotions are unchanged. SER match went from 4/12 to 6/12 (excited: happy on s1 and s3, neutral on s2). One sentence per variant is a weak signal, so the by-ear check is what counts.

### IndexTTS: strength sweep (`emo_alpha`)

IndexTTS multiplies its emotion vector by `emo_alpha` (0 to 1). Same four emotions at three lower strengths, same neutral reference voice:

| Folder | emo_alpha | SER match | s2 length: neutral / sad / excited / angry | SER notes |
|---|---|---|---|---|
| `out/indextts/` | 1.0 (round 1) | 7/12 | 4.23 / 6.03 / 4.79 / 4.23 s | neutral read as angry 2/3 |
| `out/indextts_a70/` | 0.7 | 5/12 | 4.26 / 6.30 / 4.79 / 4.59 s | excited read as angry 3/3 |
| `out/indextts_a50/` | 0.5 | 6/12 | 4.35 / 5.42 / 4.46 / 4.58 s | neutral neutral 3/3, angry 3/3; sad and excited softer |
| `out/indextts_a30/` | 0.3 | 5/12 | 4.30 / 4.67 / 4.62 / 4.54 s | nearly flat |

By the numbers alone, 0.5 looks like the best candidate for "less exaggerated but still different"; Jay's ear decides.

### Round 2 listening order (sentence 2)

Paths relative to `interpreter/bakeoff/`. Each line: neutral, then sad, then excited, then angry (`s2__<emotion>.wav` in that folder).

1. CosyVoice, new mapping: `out/cosyvoice/`
2. CosyVoice, old mapping (only excited differs): `out/cosyvoice_v1/s2__excited.wav`
3. Other CosyVoice excited variants, if the new one is not right: `out/ct_<variant>/s2__excited.wav`, with `<variant>` from the table above
4. IndexTTS at 0.5: `out/indextts_a50/`
5. IndexTTS at 0.7: `out/indextts_a70/`
6. IndexTTS at 0.3: `out/indextts_a30/`
7. IndexTTS at 1.0 (round 1, the "too exaggerated" one): `out/indextts/`

To reproduce one IndexTTS strength: `PYTHONPATH=../.. venvs/indextts/bin/python engines/adapter_indextts.py 0.5`, then normalise (`python3 -c "from run import normalise, OUT; normalise(OUT/'indextts_a50')"`).
