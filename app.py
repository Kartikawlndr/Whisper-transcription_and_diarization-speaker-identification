from flask import Flask, request, jsonify
from pydub import AudioSegment
from pyannote.audio import Pipeline
import os
from tempfile import NamedTemporaryFile

# Inisialisasi Flask app
app = Flask(__name__)

# Token Hugging Face kamu
HF_TOKEN = "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"

# Load pipeline diarization
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

@app.route("/", methods=["GET"])
def home():
    return "🟢 Diarization API is running. Use POST /diarize-audio to upload audio."

@app.route("/diarize-audio", methods=["POST"])
def diarize_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['file']

    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        file.save(temp_audio.name)

        # Tambahkan jeda diam
        audio = AudioSegment.from_wav(temp_audio.name)
        spacer = AudioSegment.silent(duration=2000)
        audio = spacer.append(audio, crossfade=0)
        audio.export(temp_audio.name, format="wav")

        # Jalankan diarization
        dz = pipeline(temp_audio.name)

        # Format hasil diarization
        results = []
        for turn, _, speaker in dz.itertracks(yield_label=True):
            results.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2)
            })

    os.remove(temp_audio.name)  # Hapus file setelah diproses
    return jsonify({"segments": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
