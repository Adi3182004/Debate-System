from faster_whisper import WhisperModel

model = WhisperModel("base", compute_type="float32")

def transcribe_audio(audio_path):
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        language=None,
        condition_on_previous_text=False
    )

    text = " ".join([seg.text for seg in segments]).strip()
    language = info.language

    return text, language