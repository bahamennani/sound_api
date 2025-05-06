from flask import Flask, request, jsonify
from utils.audio_processing import preprocess_audio
from utils.predict import predict_fault

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files or 'engine_type' not in request.form:
        return jsonify({'error': 'Audio file or engine_type not provided'}), 400

    engine_type = request.form['engine_type']
    audio_file = request.files['audio']

    spectrogram = preprocess_audio(audio_file)
    label, confidence = predict_fault(engine_type, spectrogram)

    return jsonify({'label': label, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run()