from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from flask_socketio import SocketIO

from src.config.settings import Config
from src.core.strategies.base import TranscriptionStrategy


class DeepgramStrategy(TranscriptionStrategy):
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.dg_connection = None
        self.recording = False
        self.deepgram = DeepgramClient(Config.DEEPGRAM_API_KEY)

    def on_message(self, client, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return

        self.socketio.emit('transcription', {
            'text': sentence,
            'is_final': result.is_final,
            'model': Config.DEFAULT_MODEL
        })
        print(f"Transcription: {sentence}")

    @staticmethod
    def on_error(client, error, **kwargs):
        print(f"Deepgram Error: {error}")

    def start(self):
        options = LiveOptions(
            model=Config.DEFAULT_MODEL,
            language=Config.DEFAULT_LANGUAGE,
            encoding=Config.AUDIO_ENCODING,
            endpointing=Config.ENDPOINTING_MS,
            smart_format=True,
            interim_results=True,
            punctuate=True,
            numerals=True
        )

        self.dg_connection = self.deepgram.listen.websocket.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)

        print(f"Tentando conectar com modelo: {Config.DEFAULT_MODEL}")
        print(f"Idioma: {Config.DEFAULT_LANGUAGE}")

        if not self.dg_connection.start(options):
            raise Exception("Failed to connect to Deepgram")

        self.recording = True
        print("✅ Deepgram conectado com sucesso!")

    def stop(self):
        if self.dg_connection:
            self.dg_connection.finish()
            self.dg_connection = None
        self.recording = False

    def send_audio(self, data):
        if self.recording and self.dg_connection:
            self.dg_connection.send(data)
