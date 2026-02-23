import os
import time
import queue
import sounddevice as sd
import numpy as np
import webrtcvad
import soundfile as sf
from datetime import datetime

from core.speaker_id import identify_speaker
from core.transcribe import transcribe_audio

SAMPLE_RATE = 16000
FRAME_MS = 30
VAD_MODE = 0
LOG_DIR = "data/logs"
USER_NOTE_DIR = "data/user_notes"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(USER_NOTE_DIR, exist_ok=True)

vad = webrtcvad.Vad(VAD_MODE)
audio_queue = queue.Queue()

ALLOWED_LANGS = {"en", "hi", "mr"}


def audio_callback(indata, frames, time_info, status):
    boosted = indata * 4.0
    boosted = np.clip(boosted, -1.0, 1.0)
    audio_queue.put(boosted.copy())


def save_user_note(user, text):
    user_dir = os.path.join(USER_NOTE_DIR, user)
    os.makedirs(user_dir, exist_ok=True)

    note_file = os.path.join(user_dir, "notes.txt")
    timestamp = datetime.now().strftime("%H:%M:%S")

    with open(note_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


def clear_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def record_until_silence():
    print("🎤 Listening...")

    clear_audio_queue()

    frame_samples = int(SAMPLE_RATE * FRAME_MS / 1000)

    voiced_frames = []
    silence_counter = 0
    silence_limit = 50
    speech_started = False

    max_wait_frames = 500
    wait_counter = 0

    buffer = np.array([], dtype=np.float32)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
    ):
        while True:
            chunk = audio_queue.get().flatten()
            buffer = np.concatenate((buffer, chunk))

            while len(buffer) >= frame_samples:
                frame = buffer[:frame_samples]
                buffer = buffer[frame_samples:]

                pcm16 = (frame * 32768).astype(np.int16).tobytes()

                try:
                    is_speech = vad.is_speech(pcm16, SAMPLE_RATE)
                except:
                    is_speech = False

                if not speech_started:
                    wait_counter += 1

                    if is_speech:
                        speech_started = True
                        voiced_frames.append(frame.copy())
                        continue

                    if wait_counter > max_wait_frames:
                        return None

                    continue

                if is_speech:
                    voiced_frames.append(frame.copy())
                    silence_counter = 0
                else:
                    silence_counter += 1
                    voiced_frames.append(frame.copy())

                if silence_counter > silence_limit:
                    break

            if speech_started and silence_counter > silence_limit:
                break

    if len(voiced_frames) < 8:
     return None

    audio_data = np.concatenate(voiced_frames, axis=0)
    filename = "temp_chunk.wav"
    sf.write(filename, audio_data, SAMPLE_RATE)

    return filename


def debate_mode():
    print("Debate mode started. Press Ctrl+C to stop.")
    log_file = os.path.join(LOG_DIR, f"debate_{int(time.time())}.txt")

    try:
        while True:
            audio_path = record_until_silence()
            if not audio_path:
                continue

            speaker = identify_speaker(audio_path)
            if speaker == "Unknown":
                continue

            text, lang = transcribe_audio(audio_path)

            if lang not in ALLOWED_LANGS:
                continue

            if not text or len(text.strip()) < 5:
                continue

            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] ({lang}) {speaker}: {text}"
            print(line)

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            save_user_note(speaker, text)

    except KeyboardInterrupt:
        print("\nDebate stopped cleanly.")


if __name__ == "__main__":
    debate_mode()