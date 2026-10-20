# What the AI server's TTS handler must do to give the speech emotion

Status on 2026-10-20: the orchestrator on 172.24.47.255:8791 (`ai_orchestrator.py --tts-model indextts`) answers TTS requests but
**does not use the emotion fields**. It accepts nonsense for them (`emo_vector: "banana"`, an `emo_vector` of 3 numbers,
`emo_alpha: 5`) and simply echoes the request `parameters` back in the reply, so the reply's `engine: "indextts"` is only our own
request field coming back. The audio is IndexTTS with its default emotion: measured on one sentence, "sad" was 0.97x the length of
"neutral" (a working IndexTTS gives about 1.5x), and the strength setting had no effect. Listening agrees ("no emotion felt").

## The request (unchanged; this is what the client already sends)

```json
{"robot_id": "...", "request_id": "<uuid>", "command": ["TTS"],
 "parameters": {"voice": "af_heart", "emo_vector": [0,0,0.8,0,0,0.2,0,0], "emo_alpha": 0.7, "lang": "EN"},
 "content": "I'm sorry, the meeting has been moved to next week."}
```

- `emo_vector`: exactly 8 numbers, each 0..1, in the order IndexTTS uses: `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.
- `emo_alpha`: one number, 0..1. It scales the whole vector.
- `lang`: `"EN"`. `voice` can be ignored (IndexTTS clones the reference clip instead).
- `content` may contain pronunciation hints in IndexTTS's own format, `<whānau|F AA1 . N AW0>`. They must reach IndexTTS intact (see chunking).

## What the handler must do

1. **Validate** the three emotion fields as above. If they are invalid, reply `"command": ["ERR"]` with a message in `content`
   (do not synthesize, do not echo them back as if accepted).
2. **Pass them to IndexTTS-2.5.** This is the whole fix. With `tts = IndexTTS2(cfg_path=..., model_dir=..., use_bf16=True)`
   from `indextts.infer_v2_5`:

   ```python
   torch.manual_seed(0)                       # fixed seed: the same request gives the same audio
   sr, wav = tts.infer(spk_audio_prompt=REFERENCE_WAV, text=text, output_path=None, lang=lang,
                       emo_vector=vector, emo_alpha=alpha, use_random=False)
   ```

   Do not pass `emo_audio_prompt` or `use_emo_text` (they override the vector). `REFERENCE_WAV` is a clean neutral clip of the voice to
   clone (our reference: `interpreter/local_tts/assets/neutral_ref.wav`, 3.3 s, 24 kHz).
3. **Do not split `<word|phonemes>` hints.** On GPUs under 10 GB IndexTTS splits long text at every full stop, including the dots
   inside a hint, and then reads the stress digits aloud. Either keep `tts.low_vram = False` or split only at punctuation outside `<...>`.
4. **Reply** with the usual base64 WAV (mono, 16-bit) and `parameters` `format`, `sample_rate`, `channels`, `dtype`. Put `engine`
   in the reply only as a truthful statement of what spoke, never by copying the request's parameters.

A complete working reference (HTTP and this exact protocol, one loaded model, a lock around synthesis, hint-safe chunking) is
`interpreter/local_tts/indextts_server.py` in the sign-language-interpreter repo; its `orchestrator_reply()` and `synth()` are the
two functions to compare against. It can also be run as-is beside the orchestrator on another port:

```bash
python indextts_server.py --repo <index-tts checkout> --model-dir <IndexTTS-2.5 weights> --reference <neutral_ref.wav> \
    --host 0.0.0.0 --port 8790 --ai-host 0.0.0.0 --ai-port 7900
```

## How to check it (run from any machine that can reach the server)

```bash
python3 -m interpreter.ai_server.check --host 172.24.47.255 --port <port>
```

A correct handler prints `ALL CHECKS PASSED`: it answers, refuses the three invalid settings, and "sad" averages at least 1.2x the
length of "neutral" over repeats. Our own service scores 1.42x. The orchestrator on 8791 currently fails 4 of 5 checks (sad = 0.97x).
