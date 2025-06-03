import asyncio
import threading
import json

from google import genai
from google.genai import types
from flask_socketio import SocketIO

from src.config.settings import Config
from src.core.strategies.base import TranscriptionStrategy


class GeminiStrategy(TranscriptionStrategy):
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.session = None
        self.recording = False
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = "gemini-2.0-flash-live-001"
        self.loop = None
        self.thread = None
        self.audio_buffer = []

    async def _async_start(self):
        try:
            # Configuração otimizada para transcrição usando tipos apropriados
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],
                input_audio_transcription={},  # Habilita transcrição automática
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        disabled=False,  # Usar VAD automático
                        end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                        silence_duration_ms=800  # Aguardar um pouco mais antes de finalizar
                    )
                ),
                generation_config=types.GenerationConfig(
                    temperature=0.0,  # Determinístico para transcrição
                    candidate_count=1
                )
            )

            print(f"🔧 Iniciando conexão com modelo: {self.model}")
            print(f"🔧 Configuração: input_audio_transcription habilitado, temperature=0.0")

            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                self.session = session
                self.recording = True
                print("✅ Gemini Live API conectada com sucesso")

                # Enviar áudio em buffer se existir
                if self.audio_buffer:
                    print(f"📤 Enviando {len(self.audio_buffer)} chunks em buffer")
                    for audio_data in self.audio_buffer:
                        await self._send_audio_async(audio_data)
                    self.audio_buffer = []

                print("🔄 Iniciando loop de recebimento de mensagens...")

                # Loop principal de recebimento de mensagens
                try:
                    message_count = 0
                    async for message in session.receive():
                        if not self.recording:
                            print("🛑 Parando recebimento - recording=False")
                            break

                        message_count += 1
                        print(f"📨 Mensagem #{message_count} recebida")
                        await self._process_message(message)

                        # Log periódico para mostrar que está ativo
                        if message_count % 10 == 0:
                            print(f"📊 Processadas {message_count} mensagens até agora")

                except Exception as receive_error:
                    if "ConnectionClosed" not in str(receive_error):
                        print(f"❌ Erro no loop de recebimento: {receive_error}")
                        import traceback
                        print(f"🔍 Traceback do loop: {traceback.format_exc()}")
                    else:
                        print("🔌 Conexão WebSocket fechada pelo servidor")

        except Exception as e:
            if "ConnectionClosed" not in str(e):
                print(f"❌ Erro na sessão Gemini: {e}")
                import traceback
                print(f"🔍 Traceback completo: {traceback.format_exc()}")
            else:
                print("🔌 Sessão Gemini encerrada pelo servidor")
            self.recording = False
        finally:
            self.session = None
            print("🔌 Sessão Gemini finalizada")

    async def _process_message(self, message):
        """Processa mensagens recebidas do Gemini"""
        try:
            transcription_found = False

            print(f"🔍 Processando mensagem do tipo: {type(message).__name__}")

            # Verificar server_content (padrão do Live API)
            if hasattr(message, 'server_content') and message.server_content:
                server_content = message.server_content
                print(f"✅ server_content encontrado: {type(server_content).__name__}")

                # FOCO: Input transcription (transcrição do que o usuário falou)
                if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                    if hasattr(server_content.input_transcription, 'text') and server_content.input_transcription.text:
                        text = server_content.input_transcription.text.strip()
                        if text:  # Só processa se não for vazio
                            print(f"🎤 INPUT transcription: '{text}'")
                            self._emit_transcription(text, True)
                            transcription_found = True
                    else:
                        print(f"🔍 input_transcription existe mas sem texto válido")
                else:
                    print(f"🔍 input_transcription: None ou vazio")

                # Log apenas para debug - não usar model_turn para transcrição
                if hasattr(server_content, 'model_turn') and server_content.model_turn:
                    print(f"🤖 Model turn ignorado (não é transcrição): {type(server_content.model_turn).__name__}")

            if not transcription_found:
                print(f"⚠️  Nenhuma transcrição de input encontrada")

        except Exception as e:
            print(f"❌ Erro ao processar mensagem: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")

    def _emit_transcription(self, text: str, is_final: bool):
        """Emite transcrição para o frontend"""
        try:
            self.socketio.emit('transcription', {
                'text': text,
                'is_final': is_final,
                'source': 'gemini'
            })
            print(f"📤 Transcrição enviada para frontend: {text}")
        except Exception as e:
            print(f"❌ Erro ao emitir transcrição: {e}")

    async def _send_audio_async(self, audio_data):
        """Envia áudio de forma assíncrona usando send_realtime_input"""
        try:
            if self.session:
                # Converter para bytes se necessário
                if isinstance(audio_data, (list, tuple)):
                    audio_bytes = bytes(audio_data)
                elif isinstance(audio_data, (bytes, bytearray)):
                    audio_bytes = bytes(audio_data)
                else:
                    print(f"⚠️  Tipo de áudio não reconhecido: {type(audio_data)}")
                    return

                # Log apenas primeiros chunks para evitar spam
                if len(self.audio_buffer) < 3:
                    print(f"📤 Enviando áudio: {len(audio_bytes)} bytes (VAD automático ativo)")

                # Método correto com VAD automático
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_bytes,
                        mime_type="audio/pcm;rate=16000"
                    )
                )

        except Exception as e:
            # Se for erro de conexão, só marcar sessão como None
            if "ConnectionClosed" in str(e):
                print(f"🔌 Conexão fechada durante envio de áudio")
                self.session = None
            else:
                print(f"❌ Erro ao enviar áudio: {e}")
                import traceback
                print(f"🔍 Traceback envio: {traceback.format_exc()}")

    def _run_async_loop(self):
        """Executa o loop assíncrono em thread separada"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            print("🔄 Loop assíncrono iniciado")
            self.loop.run_until_complete(self._async_start())
        except Exception as e:
            print(f"❌ Erro no loop assíncrono: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
        finally:
            if self.loop:
                self.loop.close()
            print("🔄 Loop assíncrono finalizado")

    def start(self):
        """Inicia a estratégia Gemini"""
        try:
            print("🚀 Iniciando GeminiStrategy...")
            # Reset do flag de log
            if hasattr(self, '_logged_inactive_attempt'):
                delattr(self, '_logged_inactive_attempt')

            self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.thread.start()
            print("✅ Thread iniciada com sucesso")
        except Exception as e:
            print(f"❌ Falhou ao iniciar Gemini: {e}")
            raise Exception(f"Falhou ao conectar no Gemini: {e}")

    def stop(self):
        """Para a estratégia Gemini"""
        print("🛑 Parando GeminiStrategy...")
        self.recording = False
        # Reset do flag de log
        if hasattr(self, '_logged_inactive_attempt'):
            delattr(self, '_logged_inactive_attempt')

    def send_audio(self, audio_data):
        """Envia dados de áudio"""
        if self.session and self.loop and self.recording:
            try:
                # Agendar envio no loop assíncrono
                future = asyncio.run_coroutine_threadsafe(
                    self._send_audio_async(audio_data),
                    self.loop
                )
                # Não esperar o resultado para manter fluxo

            except Exception as e:
                print(f"❌ Erro ao agendar envio de áudio: {e}")
        elif self.recording:
            # Adicionar ao buffer se ainda não conectado
            self.audio_buffer.append(audio_data)
            if len(self.audio_buffer) <= 3:  # Log apenas primeiros
                print(f"📋 Áudio adicionado ao buffer (total: {len(self.audio_buffer)}) - aguardando conexão")
        else:
            # Log apenas primeira tentativa para evitar spam
            if not hasattr(self, '_logged_inactive_attempt'):
                print("⚠️  Tentativa de envio com gravação inativa")
                self._logged_inactive_attempt = True
