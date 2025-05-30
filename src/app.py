from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from config.settings import Config
from api.routes import register_routes
from core.context import TranscriptionContext
from core.strategies.deepgram import DeepgramStrategy


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    CORS(app, resources={r"/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*")

    transcription_context = TranscriptionContext(DeepgramStrategy(socketio))

    register_routes(app, transcription_context)

    @socketio.on('audio_data')
    def handle_audio_data(data):
        transcription_context.send_audio(data)

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        socketio.emit('status', {'connected': True})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
