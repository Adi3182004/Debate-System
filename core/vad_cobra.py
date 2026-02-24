import pvcobra
import sounddevice as sd
import numpy as np

ACCESS_KEY = "Access_key_from_picovoice"

cobra = pvcobra.create(access_key=ACCESS_KEY)

SAMPLE_RATE = cobra.sample_rate
FRAME_LENGTH = cobra.frame_length

SELECTED_DEVICE = None


def auto_select_input_device():
    global SELECTED_DEVICE
    try:
        default_device = sd.default.device[0]
        info = sd.query_devices(default_device, "input")
        SELECTED_DEVICE = default_device
        print(f"🎙️ Using mic: {info['name']}")
    except Exception:
        info = sd.query_devices(kind="input")
        print(f"🎙️ Using mic: {info['name']}")
        SELECTED_DEVICE = None


def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    if peak < 1:
        return audio
    audio = audio.astype(np.float32) / peak
    audio = audio * 32767
    return audio.astype(np.int16)


def smart_agc(pcm):
    energy = np.abs(pcm).mean()

    if energy < 300:
        gain = 6.0
    elif energy < 800:
        gain = 4.0
    elif energy < 2000:
        gain = 2.5
    else:
        gain = 1.5

    boosted = np.clip(pcm * gain, -32768, 32767)
    return boosted.astype(np.int16)


def record_until_silence():
    voiced_frames = []
    silence_counter = 0
    silence_limit = 65
    speech_started = False
    printed_listening = False
    last_energy_print = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=SELECTED_DEVICE,
        blocksize=FRAME_LENGTH
    ) as stream:

        while True:
            if not printed_listening:
                print("🎤 Listening...")
                printed_listening = True

            frame, _ = stream.read(FRAME_LENGTH)
            pcm = frame.flatten().astype(np.int16)

            energy_norm = np.abs(pcm).mean() / 32768

            if abs(energy_norm - last_energy_print) > 0.002:
                print(f"\r🔊 Mic level: {energy_norm:.3f}", end="")
                last_energy_print = energy_norm

            pcm_boosted = smart_agc(pcm)

            voice_prob = cobra.process(pcm_boosted)

            if not speech_started:
                if voice_prob > 0.22:
                    speech_started = True
                    print("\n🟢 Speech detected")
                    voiced_frames.append(pcm_boosted.copy())
                continue

            if voice_prob > 0.16:
                voiced_frames.append(pcm_boosted.copy())
                silence_counter = 0
            else:
                silence_counter += 1
                voiced_frames.append(pcm_boosted.copy())

            if silence_counter > silence_limit:
                break

    if not voiced_frames:
        return None

    audio = np.concatenate(voiced_frames).astype(np.int16)
    audio = normalize_audio(audio)

    return audio, SAMPLE_RATE
def print_input_device():
    auto_select_input_device()
