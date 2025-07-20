import hashlib
import wave

def calculate_audio_hash(file_path):
    CHUNK_SIZE = 4096

    with wave.open(file_path, "rb") as wf:
        audio_hash = hashlib.sha256()
        while True:

            audio_data = wf.readframes(CHUNK_SIZE)
            if not audio_data:
                break

            audio_hash.update(audio_data)

    return audio_hash.hexdigest()

wav_file1_path = "a.wav"
wav_file2_path = "a.wav"

try:
    hash1 = calculate_audio_hash(wav_file1_path)
    hash2 = calculate_audio_hash(wav_file2_path)


    if hash1 == hash2:
        print("The sound files have same audio content.")
    else:
        print("The sound files do not same audio content.")
except Exception as e:
    print("")
