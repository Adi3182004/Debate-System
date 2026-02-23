import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

BASE_DIR = "data/voices"
encoder = VoiceEncoder()


def load_voiceprints():
    voiceprints = {}

    if not os.path.exists(BASE_DIR):
        return voiceprints

    for user in os.listdir(BASE_DIR):
        vp_path = os.path.join(BASE_DIR, user, "voiceprint.npy")
        if os.path.exists(vp_path):
            emb = np.load(vp_path)
            emb = emb / np.linalg.norm(emb)
            voiceprints[user] = emb

    return voiceprints


VOICEPRINTS = load_voiceprints()


def cosine_similarity(a, b):
    return np.dot(a, b)


def identify_speaker(audio_path):
    if not VOICEPRINTS:
        return "Unknown"

    wav = preprocess_wav(audio_path)
    embed = encoder.embed_utterance(wav)
    embed = embed / np.linalg.norm(embed)

    best_user = "Unknown"
    best_score = -1.0

    for user, ref_embed in VOICEPRINTS.items():
        score = cosine_similarity(embed, ref_embed)
        if score > best_score:
            best_score = score
            best_user = user

    if best_score < 0.70:
        return "Unknown"

    return best_user