from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from faster_whisper import WhisperModel

ALLOWED_LANGS = {"en", "hi", "mr"}

model = WhisperModel(
    "small",
    compute_type="int8"
)


def to_hinglish(text, lang):
    if lang in {"hi", "mr"} and text:
        try:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        except Exception:
            return text
    return text


def transcribe_audio(audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        language=None
    )

    text_parts = []
    for seg in segments:
        if seg.text and seg.text.strip():
            text_parts.append(seg.text.strip())

    text = " ".join(text_parts).strip()
    language = info.language

    if language not in ALLOWED_LANGS:
        language = "unknown"

    text = to_hinglish(text, language)

    return text, language