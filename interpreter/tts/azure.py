"""Azure Speech SSML builder. String only; no Azure SDK, no network, no keys."""
from typing import List, Optional

from interpreter.contracts import SpeechPlan
from interpreter.tts.base import Rendered, TTSConfig, add_note, clamp, escape_xml, ssml_body


def render(plan: SpeechPlan, cfg: Optional[TTSConfig] = None) -> Rendered:
    cfg = cfg or TTSConfig()
    notes: List[str] = ["styles (express-as) not used; support for en-NZ voices unconfirmed"]
    p = plan.prosody
    voice = plan.voice.voice_id or cfg.azure_voice

    rate = clamp(p.rate, 0.5, 2, notes, "rate")
    pitch = clamp((p.pitch - 1) * 100, -50, 50, notes, "pitch%")
    volume = (p.volume - 1) * 100
    if p.emphasis:
        add_note(notes, "emphasis dropped: <emphasis> is documented for only three en-US voices")

    payload = ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-NZ">'
               f'<voice name="{escape_xml(voice)}">'
               f'<prosody rate="{rate:.2f}" pitch="{pitch:+.0f}%" volume="{volume:+.0f}%">'
               f'{ssml_body(plan, notes)}</prosody></voice></speak>')
    return Rendered("azure", payload, {"voice": voice}, tuple(notes))
