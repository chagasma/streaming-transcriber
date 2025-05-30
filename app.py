import os
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

from context.transcript_context import TranscriptionContext
from strategies.deepgram_strategy import DeepgramStrategy

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
socketio = SocketIO(app, cors_allowed_origins="*")

transcription_context = TranscriptionContext(DeepgramStrategy(socketio))


@app.route('/')
def serve_html():
    return send_file('test.html')


@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        transcription_context.start()
        return jsonify({"message": "Recording started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    transcription_context.stop()
    return jsonify({"message": "Recording stopped"})


@socketio.on('audio_data')
def handle_audio_data(data):
    transcription_context.send_audio(data)


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'connected': True})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
