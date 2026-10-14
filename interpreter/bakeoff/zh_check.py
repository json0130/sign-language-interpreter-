"""Verify Mandarin support: `render <engine>` (run in that engine's venv) writes out/<engine>/zh/zh_check.wav;
`asr` (run in venvs/cosyvoice, which has openai-whisper) transcribes each and compares with the source text.
A smoke check that the model produces recognisable Mandarin, not a quality rating."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HERE, OUT, save_wav  # noqa: E402

ZH_TEXT = "我的家人明天来开会。"   # "My family is coming to the meeting tomorrow."


def render(engine: str) -> None:
    import torch
    (OUT / engine / "zh").mkdir(parents=True, exist_ok=True)
    if engine == "chatterbox":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        wav, sr = m.generate(ZH_TEXT, language_id="zh"), m.sr
    elif engine == "cosyvoice":
        sys.path[:0] = [str(HERE / "repos/CosyVoice"), str(HERE / "repos/CosyVoice/third_party/Matcha-TTS")]
        from cosyvoice.cli.cosyvoice import AutoModel
        m = AutoModel(model_dir=str(HERE / "models/cosyvoice3"))
        parts = [o["tts_speech"] for o in m.inference_instruct2(
            ZH_TEXT, "You are a helpful assistant.<|endofprompt|>", str(HERE / "repos/CosyVoice/asset/zero_shot_prompt.wav"))]
        wav, sr = torch.cat(parts, dim=1), m.sample_rate
    elif engine == "indextts":
        import numpy as np
        sys.path.insert(0, str(HERE / "repos/index-tts"))
        from indextts.infer_v2_5 import IndexTTS2
        d = HERE / "models/indextts-2.5"
        m = IndexTTS2(cfg_path=str(d / "config.yaml"), model_dir=str(d), use_bf16=True)
        sr, w = m.infer(spk_audio_prompt=str(HERE / "repos/index-tts/examples/voice_01.wav"), text=ZH_TEXT,
                        output_path=None, lang="ZH", use_random=False)
        wav = np.asarray(w, dtype="float32").reshape(-1) / 32768.0
    else:
        raise SystemExit(f"no Mandarin path for {engine}")
    print(engine, "zh seconds:", round(save_wav(OUT / engine / "zh" / "zh_check.wav", wav, sr), 2))


def asr() -> None:
    import whisper
    model = whisper.load_model("small", download_root=str(HERE / "models" / "whisper"))
    result = {}
    for wav in sorted(OUT.glob("*/zh/zh_check.wav")):
        text = model.transcribe(str(wav), language="zh", fp16=True)["text"].strip()
        overlap = len(set(text) & set(ZH_TEXT)) / len(set(ZH_TEXT))
        result[wav.parts[-3]] = {"transcript": text, "char_overlap": round(overlap, 2), "exact": text.strip("。 ，,.") == ZH_TEXT.strip("。")}
        print(wav.parts[-3], result[wav.parts[-3]], flush=True)
    (OUT / "zh_asr.json").write_text(json.dumps({"source_text": ZH_TEXT, "asr_model": "openai-whisper small", "results": result},
                                                indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asr() if sys.argv[1] == "asr" else render(sys.argv[1])
