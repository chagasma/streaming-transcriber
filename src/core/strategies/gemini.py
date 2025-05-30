import asyncio
import threading
import os
from google import genai
from google.genai import types
from flask_socketio import SocketIO

from src.core.strategies.base import TranscriptionStrategy


class GeminiStrategy(TranscriptionStrategy):
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.session = None
        self.recording = False
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = "gemini-2.0-flash-live-001"
        self.loop = None
        self.thread = None

    async def _async_start(self):
        try:
            config = {
                "response_modalities": ["TEXT"],
                "realtime_input_config": {
                    "automatic_activity_detection": {
                        "disabled": False,
                        "start_of_speech_sensitivity": types.StartSensitivity.START_SENSITIVITY_MEDIUM,
                        "end_of_speech_sensitivity": types.EndSensitivity.END_SENSITIVITY_MEDIUM
                    }
                }
            }

            self.session = await self.client.aio.live.connect(model=self.model, config=config)
            self.recording = True

            async for response in self.session.receive():
                if not self.recording:
                    break

                if response.text is not None:
                    self.socketio.emit('transcription', {
                        'text': response.text,
                        'is_final': True
                    })
                    print(f"Gemini Transcription: {response.text}")

        except Exception as e:
            print(f"Erro na sessão Gemini: {e}")
            self.recording = False

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
        if self.session and self.loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.session.aclose(), self.loop
                )
                future.result(timeout=2.0)
            except:
                pass

    def send_audio(self, audio_data):
        if self.recording and self.session and self.loop:
            try:
                audio_bytes = bytes(audio_data)
                audio_blob = types.Blob(
                    data=audio_bytes,
                    mime_type="audio/pcm;rate=16000"
                )

                asyncio.run_coroutine_threadsafe(
                    self.session.send_realtime_input(audio=audio_blob),
                    self.loop
                )

            except Exception as e:
                print(f"Erro ao enviar áudio Gemini: {e}")
