import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

BASE_DIR = "data/voices"
encoder = VoiceEncoder()

UNKNOWN_THRESHOLD = 0.66
MARGIN_THRESHOLD = 0.035
MIN_AUDIO_SEC = 1.2


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

    if len(wav) < 16000 * MIN_AUDIO_SEC:
        return "Unknown"

    embed = encoder.embed_utterance(wav)
    embed = embed / np.linalg.norm(embed)

    scores = []

    for user, ref_embed in VOICEPRINTS.items():
        score = cosine_similarity(embed, ref_embed)
        scores.append((user, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    best_user, best_score = scores[0]

    if len(scores) > 1:
        second_score = scores[1][1]
        if best_score - second_score < MARGIN_THRESHOLD:
            return "Unknown"

    if best_score < UNKNOWN_THRESHOLD:
        return "Unknown"

    return best_user