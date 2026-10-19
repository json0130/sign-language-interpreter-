"""SpeechPlan -> WAV file with local prosody. Standard library only (wave, array)."""
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

from interpreter.contracts import SpeechPlan
from interpreter.emotion import CANONICAL_VA
from interpreter.local_tts.config import RenderConfig
from interpreter.local_tts.engines import engine_for
from interpreter.tts.base import lexicon_spans, pause_map, split_punct, tokens

PITCH_RANGE = (0.8, 1.25)
GAP_MS = 120          # silence between clauses when a plan with segments is joined into one WAV
GAIN_RANGE = (0.25, 1.5)
INT16_MAX = 32767


@dataclass(frozen=True)
class RenderResult:
    path: str
    duration_s: float
    sample_rate: int
    params: dict
    notes: Tuple[str, ...]


def _clamp(value: float, lo: float, hi: float, label: str, notes: List[str]) -> float:
    clipped = max(lo, min(hi, value))
    if clipped != value:
        notes.append(f"{label} clamped from {value:g} to {clipped:g}")
    return clipped


def respelled_segments(plan: SpeechPlan, style: str) -> List[Tuple[str, int]]:
    """[(text, pause_ms_after)] with lexicon respellings applied. Without a lexicon match the
    words equal plan.utterance. A pause inside a lexicon phrase moves to the phrase end."""
    toks = tokens(plan.utterance)
    spans = lexicon_spans(toks, plan.lexicon)
    pauses = pause_map(plan, spans)
    segs: List[Tuple[str, int]] = []
    words: List[str] = []
    i = 0
    while i < len(toks):
        end, entry = spans.get(i, (i + 1, None))
        if entry is None:
            words.append(" ".join(toks[i:end]))
        else:
            core = entry.respelling.lower()
            if style == "joined":
                core = core.replace("-", "")
            words.append(split_punct(toks[i])[0] + core + split_punct(toks[end - 1])[2])
        ms = sum(pauses.get(end - 1, []))
        if ms:
            segs.append((" ".join(words), ms))
            words = []
        i = end
    if words:
        segs.append((" ".join(words), 0))
    return segs


def render_speech(plan: SpeechPlan, engine, out_path, cfg: Optional[RenderConfig] = None) -> RenderResult:
    """Synthesise `plan` to a mono 16-bit WAV at `out_path`. `engine=None` picks the engine for
    plan.voice.language (engine_for); an explicit engine wins.

    A plan with segments (the emotion flow) is rendered one clause at a time, each with its own prosody, and the
    clauses are joined with GAP_MS of silence (see _render_one for how one clause is rendered). The result's params
    are the first clause's, plus params["segments"] with every clause's params and duration.

    An engine with `emotional = True` (IndexTTS) is given each clause's emotion instead (_render_emotional): it does
    its own expressive delivery, so our rate/pitch trick is not applied; only the clause volume is.
    """
    cfg = cfg or RenderConfig()
    if engine is None:
        engine = engine_for(plan.voice.language)
    emotional = getattr(engine, "emotional", False)
    if emotional:
        from interpreter.local_tts.indextts_emotion import FLOW_STRENGTH
        flow_scale = FLOW_STRENGTH if len({s.emotion_label for s in plan.segments}) > 1 else 1.0
    if not plan.segments:
        if emotional:
            v, a = CANONICAL_VA[plan.emotion_label]
            return _render_emotional(plan.utterance, v, a, plan.prosody.volume, engine, out_path, cfg)
        return _render_one(plan, engine, out_path, cfg)
    with tempfile.TemporaryDirectory() as d:
        parts = []
        for k, seg in enumerate(plan.segments):
            path = Path(d) / f"{k}.wav"
            if emotional:
                parts.append(_render_emotional(seg.text, seg.valence, seg.arousal, seg.prosody.volume, engine, path, cfg,
                                               flow_scale))
                continue
            part = replace(plan, utterance=seg.text, emotion_label=seg.emotion_label, prosody=seg.prosody, segments=[])
            parts.append(_render_one(part, engine, path, cfg))
        duration = join_wavs([r.path for r in parts], out_path)
    params = {**parts[0].params, "segments": [{**r.params, "duration_s": r.duration_s} for r in parts]}
    notes = tuple(dict.fromkeys(n for r in parts for n in r.notes))
    with wave.open(str(out_path), "rb") as w:
        rate = w.getframerate()
    return RenderResult(str(out_path), duration, rate, params, notes)


def resample(samples: array, src: int, dst: int) -> array:
    """Linear-interpolation resample. Each clause is written at its own rate (that is how pitch is applied), so
    clauses are brought to one rate before joining; playback sounds the same as the separate files."""
    if src == dst or not samples:
        return array("h", samples)
    n, step, last = round(len(samples) * dst / src), src / dst, len(samples) - 1
    out = array("h", bytes(2 * n))
    for i in range(n):
        x = i * step
        j = min(int(x), last)
        nxt = samples[min(j + 1, last)]
        out[i] = round(samples[j] + (nxt - samples[j]) * (x - j))
    return out


def join_wavs(paths, out_path, gap_ms: int = GAP_MS) -> float:
    """Concatenate mono 16-bit WAVs (any sample rates) with short silences between them; returns seconds."""
    loaded = []
    for p in paths:
        with wave.open(str(p), "rb") as w:
            s = array("h")
            s.frombytes(w.readframes(w.getnframes()))
            loaded.append((s, w.getframerate()))
    rate = max(r for _, r in loaded)
    joined = array("h")
    for k, (s, r) in enumerate(loaded):
        if k:
            joined.extend(array("h", bytes(2 * round(gap_ms / 1000 * rate))))
        joined.extend(resample(s, r, rate))
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(joined.tobytes())
    return len(joined) / rate


def _normalise_and_write(samples: array, rate: int, gain: float, out_path, cfg: RenderConfig) -> None:
    """Scale to cfg.base_peak, apply `gain`, clip to int16 and write a mono 16-bit WAV."""
    peak = max((abs(s) for s in samples), default=0)
    if peak:
        k = cfg.base_peak * INT16_MAX / peak * gain
        samples = array("h", (max(-INT16_MAX, min(INT16_MAX, round(s * k))) for s in samples))
    if sys.byteorder == "big":   # WAV data is little-endian
        samples.byteswap()
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def trim_silence(samples: array, rate: int, threshold: float = 0.02, pad_ms: int = 40) -> array:
    """Drop leading and trailing near-silence (below `threshold` x peak), keeping `pad_ms` either side."""
    peak = max((abs(x) for x in samples), default=0)
    if not peak:
        return samples
    limit = peak * threshold
    loud = [i for i, x in enumerate(samples) if abs(x) > limit]
    pad = round(rate * pad_ms / 1000)
    return samples[max(0, loud[0] - pad):min(len(samples), loud[-1] + pad + 1)]


def _render_emotional(text: str, valence: float, arousal: float, volume: float, engine, out_path,
                      cfg: RenderConfig, alpha_scale: float = 1.0) -> RenderResult:
    """One clause on an emotion-capable engine: the clause's emotion becomes the engine's emotion vector and alpha
    (times `alpha_scale`, which softens sentences whose emotion changes). Kupu get phoneme hints if configured."""
    from interpreter.local_tts.config import INDEXTTS_KUPU_HINTS
    from interpreter.local_tts.indextts_emotion import emotion_vector
    from interpreter.local_tts.indextts_phonemes import annotate
    notes: List[str] = []
    gain = _clamp(volume, *GAIN_RANGE, "volume", notes)
    vector, alpha = emotion_vector(valence, arousal)
    alpha = round(alpha * alpha_scale, 3)
    sent = annotate(text) if INDEXTTS_KUPU_HINTS else text
    pcm, rate = engine.synthesize_emotion(sent, vector, alpha)
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    samples = trim_silence(samples, rate)     # IndexTTS pads each clause; joined clauses had awkward gaps
    _normalise_and_write(samples, rate, gain, out_path, cfg)
    params = {"rate": 1.0, "pitch": 1.0, "volume": gain, "emo_vector": vector, "emo_alpha": alpha,
              "flow_softened": alpha_scale != 1.0, "sent_text": sent,
              "native_sample_rate": rate, "output_sample_rate": rate}
    return RenderResult(str(out_path), len(samples) / rate, rate, params, tuple(notes))


def _render_one(plan: SpeechPlan, engine, out_path, cfg: RenderConfig) -> RenderResult:
    """One delivery (plan.prosody) for the whole text.

    Rate and pitch from one engine and one relabel. With r = prosody.rate and p = prosody.pitch
    (p clamped to 0.8..1.25):
      * synthesise with length_scale = base * p / r  (the voice's own pace stretched by p/r);
      * label the WAV with sample rate round(native * p).
    Playing N samples at p times the native rate shortens them by 1/p and raises pitch by p, and
    that also speeds tempo by p. The length_scale already stretched the audio by p, so the two
    cancel on tempo: net duration = base_duration / r, i.e. tempo r. Pitch is p.

    Pauses are silence samples sized against the OUTPUT rate (ms/1000 * round(native * p)), so
    they last the stated milliseconds in real playback, not in native-rate time.

    Volume: the whole waveform is first normalised to cfg.base_peak, then multiplied by
    prosody.volume (clamped 0.25..1.5) and clipped to int16. Emphasis and the plan's eye colour
    are not used here.
    """
    notes: List[str] = []
    pr = plan.prosody
    rate = pr.rate
    pitch = _clamp(pr.pitch, *PITCH_RANGE, "pitch", notes)
    gain = _clamp(pr.volume, *GAIN_RANGE, "volume", notes)
    if pr.emphasis:
        notes.append("emphasis not rendered by local backend")

    length_scale = engine.base_length_scale * pitch / rate
    out_rate = round(engine.native_sample_rate * pitch)

    samples = array("h")
    for text, pause_ms in respelled_segments(plan, cfg.respell_style):
        seg = array("h")
        seg.frombytes(engine.synthesize(text, length_scale))
        samples.extend(seg)
        samples.extend(array("h", bytes(2 * round(pause_ms / 1000 * out_rate))))

    _normalise_and_write(samples, out_rate, gain, out_path, cfg)

    params = {"rate": rate, "pitch": pitch, "volume": gain, "length_scale": length_scale,
              "native_sample_rate": engine.native_sample_rate, "output_sample_rate": out_rate}
    return RenderResult(str(out_path), len(samples) / out_rate, out_rate, params, tuple(notes))
