import asyncio
import threading

from google import genai
from flask_socketio import SocketIO

from src.config.settings import Config
from src.core.strategies.base import TranscriptionStrategy


class GeminiStrategy(TranscriptionStrategy):
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.session = None
        self.recording = False
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = "gemini-2.0-flash-live-preview-04-09"
        self.loop = None
        self.thread = None
        self.audio_buffer = []

    async def _async_start(self):
        try:
            config = {
                "response_modalities": ["TEXT"],
                "input_audio_transcription": {}
            }

            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                self.session = session
                self.recording = True
                print("Gemini Live API conectada com sucesso")

                if self.audio_buffer:
                    for audio_data in self.audio_buffer:
                        await self._send_audio_async(audio_data)
                    self.audio_buffer = []

                async for message in session.receive():
                    if not self.recording:
                        break

                    try:
                        if hasattr(message, 'text') and message.text:
                            self.socketio.emit('transcription', {
                                'text': message.text,
                                'is_final': True
                            })
                            print(f"Gemini Transcription: {message.text}")

                        if hasattr(message, 'server_content') and message.server_content:
                            # Log para debug
                            if hasattr(message.server_content, 'input_transcription'):
                                print("Input transcription received:", message.server_content.input_transcription)

                            if hasattr(message.server_content, 'model_turn') and message.server_content.model_turn:
                                for part in message.server_content.model_turn.parts:
                                    if hasattr(part, 'text') and part.text:
                                        self.socketio.emit('transcription', {
                                            'text': part.text,
                                            'is_final': True
                                        })
                                        print(f"Gemini Model Turn: {part.text}")

                    except Exception as e:
                        print(f"Erro ao processar mensagem: {e}")

        except Exception as e:
            print(f"Erro na sessão Gemini: {e}")
            self.recording = False
        finally:
            self.session = None

    async def _send_audio_async(self, audio_data):
        try:
            if self.session:
                audio_bytes = bytes(audio_data)
                await self.session.send_realtime_input(
                    audio={
                        "data": audio_bytes,
                        "mime_type": "audio/pcm;rate=16000"
                    }
                )
        except Exception as e:
            print(f"Erro ao enviar áudio: {e}")

    def _run_async_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._async_start())
        finally:
            self.loop.close()

    def start(self):
        try:
            self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.thread.start()
        except Exception as e:
            raise Exception(f"Falhou ao conectar no Gemini: {e}")

    def stop(self):
        self.recording = False

    def send_audio(self, audio_data):
        if self.session and self.loop and self.recording:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_audio_async(audio_data),
                    self.loop
                )
            except Exception as e:
                print(f"Erro ao agendar envio de áudio: {e}")
        elif self.recording:
            self.audio_buffer.append(audio_data)
