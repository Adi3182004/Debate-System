import os
import time
import soundfile as sf
from datetime import datetime

from core.speaker_id import identify_speaker
from core.transcribe import transcribe_audio
from core.vad_cobra import record_until_silence, auto_select_input_device
LOG_DIR = "data/logs"
USER_NOTE_DIR = "data/user_notes"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(USER_NOTE_DIR, exist_ok=True)

ALLOWED_LANGS = {"en", "hi", "mr"}


def save_user_note(user, text):
    user_dir = os.path.join(USER_NOTE_DIR, user)
    os.makedirs(user_dir, exist_ok=True)

    note_file = os.path.join(user_dir, "notes.txt")
    timestamp = datetime.now().strftime("%H:%M:%S")

    with open(note_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

from core.vad_cobra import auto_select_input_device
auto_select_input_device()
def debate_mode():
    print("Debate mode started. Press Ctrl+C to stop.")
    auto_select_input_device()

    log_file = os.path.join(LOG_DIR, f"debate_{int(time.time())}.txt")

    try:
        while True:
            audio_data = record_until_silence()

            if not audio_data:
                time.sleep(0.2)
                continue

            print("✅ Processing speech...\n")

            audio_array, sr = audio_data

            if audio_array is None or len(audio_array) == 0:
                continue

            temp_file = "temp_chunk.wav"
            sf.write(temp_file, audio_array, sr)

            speaker = identify_speaker(temp_file)
            if speaker == "Unknown":
                print("⚠️ Unknown speaker — ignored\n")
                continue

            text, lang = transcribe_audio(temp_file)

            if lang not in ALLOWED_LANGS:
                print(f"⚠️ Language '{lang}' ignored\n")
                continue

            if not text or len(text.strip()) < 5:
                print("⚠️ Weak transcription — ignored\n")
                continue

            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] ({lang}) {speaker}: {text}"
            print(line + "\n")

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            save_user_note(speaker, text)

    except KeyboardInterrupt:
        print("\nDebate stopped cleanly.")


if __name__ == "__main__":
    debate_mode()