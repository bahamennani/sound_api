import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os

def preprocess_audio(file_storage):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".aac") as temp_audio:
        file_storage.save(temp_audio.name)

    wav_path = temp_audio.name.replace(".aac", ".wav")
    sound = AudioSegment.from_file(temp_audio.name, format="aac")
    sound.export(wav_path, format="wav")

    y, sr = librosa.load(wav_path, sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    spectrogram_db = librosa.util.fix_length(spectrogram_db, size=128, axis=1)
    spectrogram_db = spectrogram_db[:128, :]
    spectrogram_db = spectrogram_db.astype(np.float32)
    spectrogram_db = np.expand_dims(spectrogram_db, axis=0)
    spectrogram_db = np.expand_dims(spectrogram_db, axis=-1)

    os.remove(temp_audio.name)
    os.remove(wav_path)

    return spectrogram_db