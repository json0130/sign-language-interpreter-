"""Amazon Polly SSML builder for the NEURAL engine. String only; no AWS SDK, no network, no keys."""
import math
from typing import List, Optional

from interpreter.contracts import SpeechPlan
from interpreter.tts.base import Rendered, TTSConfig, add_note, clamp, ssml_body


def render(plan: SpeechPlan, cfg: Optional[TTSConfig] = None) -> Rendered:
    cfg = cfg or TTSConfig()
    notes: List[str] = ["neural prosody support is partial in the Polly docs; "
                        "verify rate/volume are honoured in the listening test"]
    p = plan.prosody

    rate = round(clamp(round(100 * p.rate), 20, 200, notes, "rate%"))
    db = clamp(round(20 * math.log10(p.volume), 1), -6, 6, notes, "volume dB")
    attrs = f'rate="{rate}%" volume="{db:+.1f}dB"'
    if cfg.polly_use_pitch:
        attrs += f' pitch="{(p.pitch - 1) * 100:+.0f}%"'
    if p.emphasis:
        add_note(notes, "emphasis dropped: <emphasis> unavailable on Polly neural voices")

    payload = f"<speak><prosody {attrs}>{ssml_body(plan, notes)}</prosody></speak>"
    params = {"Engine": "neural", "VoiceId": plan.voice.voice_id or cfg.polly_voice,
              "LanguageCode": "en-NZ", "TextType": "ssml"}
    return Rendered("polly", payload, params, tuple(notes))
