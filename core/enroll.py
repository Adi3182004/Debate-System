import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder, preprocess_wav

SAMPLE_RATE = 16000
DURATION = 5
BASE_DIR = "data/voices"

os.makedirs(BASE_DIR, exist_ok=True)

encoder = VoiceEncoder()


def record_voice(filepath):
    print("🎤 Recording... Speak clearly")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    write(filepath, SAMPLE_RATE, audio)
    print("Saved:", filepath)


def enroll_user(username, num_samples=3):
    user_dir = os.path.join(BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)

    embeddings = []

    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}")
        filepath = os.path.join(user_dir, f"sample_{i+1}.wav")
        record_voice(filepath)

        wav = preprocess_wav(filepath)
        embed = encoder.embed_utterance(wav)
        embeddings.append(embed)

    mean_embedding = np.mean(embeddings, axis=0)
    np.save(os.path.join(user_dir, "voiceprint.npy"), mean_embedding)

    print(f"\n✅ {username} enrolled successfully with {num_samples} samples")


if __name__ == "__main__":
    name = input("Enter username: ").strip().lower()
    enroll_user(name, num_samples=3)