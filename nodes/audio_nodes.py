"""MeesaJarJar audio nodes - extracted from jarjar_audio.py + jarjar_audio_load_sample.py (Old pack)."""

import logging
import os
import uuid

import torch
import torchaudio

log = logging.getLogger("jarjar")

try:
    import folder_paths
except ImportError:
    folder_paths = None


def _default_audio_output_dir():
    try:
        if folder_paths is not None:
            return folder_paths.get_output_directory()
    except (OSError, ValueError, AttributeError):
        pass
    return ""


class JarJarAudioSaveAudio:
    """Save AUDIO dict (F5-TTS style {waveform, sample_rate}) to disk with UUID filename."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "multiline": False,
                    "default": _default_audio_output_dir()
                }),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "filename", "path")
    FUNCTION = "save_audio"
    CATEGORY = "JarJar/Audio"
    DESCRIPTION = "Saves an AUDIO dict to a UUID-named .wav file. Returns passthrough audio + filename + full path."

    def save_audio(self, file_path, audio):
        if not isinstance(audio, dict):
            raise TypeError("Expected audio input to be a dict with 'waveform' and 'sample_rate'.")

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")

        if waveform is None or sample_rate is None:
            raise ValueError("Missing waveform or sample_rate in audio input.")

        if waveform.dim() == 3 and waveform.shape[0] == 1:
            waveform = waveform[0]

        filename = f"{uuid.uuid4()}.wav"
        full_path = os.path.join(file_path, filename)
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        torchaudio.save(full_path, waveform, sample_rate)
        log.info("[JarJarAudio] Saved to %s", full_path)

        return (audio, filename, full_path)


class JarJarAudioLoadSample:
    """Load sample-NNNNNN.mp3 from a folder by index.

    Tries hyphen format first (sample-000001.mp3), falls back to dot format (sample.000001.mp3).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": ""
                }),
                "sample_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 999999
                }),
            }
        }

    CATEGORY = "JarJar/Audio"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_sample"
    DESCRIPTION = "Loads sample-<index>.mp3 from folder_path by 1-based index. Returns AUDIO dict."

    def load_sample(self, folder_path, sample_index):
        folder_path = os.path.normpath(folder_path)
        padded_index = f"{sample_index:06d}"

        filename = f"sample-{padded_index}.mp3"
        file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            filename = f"sample.{padded_index}.mp3"
            file_path = os.path.join(folder_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        waveform, sample_rate = torchaudio.load(file_path)
        return ({
            "waveform": waveform.unsqueeze(0),
            "sample_rate": sample_rate
        },)


NODE_CLASS_MAPPINGS = {
    "JarJarAudioSaveAudio": JarJarAudioSaveAudio,
    "JarJarAudioLoadSample": JarJarAudioLoadSample,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarAudioSaveAudio": "JarJar Audio Save Audio",
    "JarJarAudioLoadSample": "JarJar Audio Load Sample",
}
