from flask_socketio import SocketIO

from src.core.strategies.base import TranscriptionStrategy


class GeminiStrategy(TranscriptionStrategy):
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio

    def start(self):
        pass

    def stop(self):
        pass

    def send_audio(self, data):
        pass
