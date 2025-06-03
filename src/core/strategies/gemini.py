import asyncio
import threading
import json

from google import genai
from google.genai.types import Part
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
            # Configuração correta baseada na documentação oficial do Vertex AI
            config = {
                "response_modalities": ["TEXT"],
                "input_audio_transcription": {},  # Objeto vazio habilita transcrição de entrada
                "output_audio_transcription": {},  # Objeto vazio habilita transcrição de saída
                "generation_config": {
                    "temperature": 0.1,
                    "candidate_count": 1
                }
            }

            print(f"🔧 Iniciando conexão com modelo: {self.model}")
            print(f"🔧 Configuração: {json.dumps(config, indent=2)}")

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

                # Adicionar timeout para debug
                print("🔄 Iniciando loop de recebimento de mensagens...")

                # Loop principal de recebimento de mensagens
                try:
                    async for message in session.receive():
                        if not self.recording:
                            print("🛑 Parando recebimento - recording=False")
                            break

                        print(f"📨 Mensagem recebida: {type(message).__name__}")
                        await self._process_message(message)

                except Exception as receive_error:
                    print(f"❌ Erro no loop de recebimento: {receive_error}")
                    import traceback
                    print(f"🔍 Traceback do loop: {traceback.format_exc()}")

        except Exception as e:
            print(f"❌ Erro na sessão Gemini: {e}")
            import traceback
            print(f"🔍 Traceback completo: {traceback.format_exc()}")
            self.recording = False
        finally:
            self.session = None
            print("🔌 Sessão Gemini finalizada")

    async def _process_message(self, message):
        """Processa mensagens recebidas do Gemini de forma mais robusta"""
        try:
            transcription_found = False

            # Log detalhado da mensagem para debug
            print(f"🔍 Processando mensagem do tipo: {type(message).__name__}")

            # Verificar todos os atributos da mensagem
            if hasattr(message, '__dict__'):
                attrs = list(message.__dict__.keys())
                print(f"📋 Atributos disponíveis: {attrs}")

            # Método 1: Verificar server_content (padrão do Live API)
            if hasattr(message, 'server_content') and message.server_content:
                server_content = message.server_content
                print(f"✅ server_content encontrado: {type(server_content).__name__}")

                # Input transcription (transcrição do que o usuário falou)
                if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                    if hasattr(server_content.input_transcription, 'text'):
                        text = server_content.input_transcription.text
                        print(f"🎤 INPUT transcription: {text}")
                        self._emit_transcription(text, True)
                        transcription_found = True

                # Output transcription (transcrição do que o modelo falou)
                if hasattr(server_content, 'output_transcription') and server_content.output_transcription:
                    if hasattr(server_content.output_transcription, 'text'):
                        text = server_content.output_transcription.text
                        print(f"🔊 OUTPUT transcription: {text}")
                        self._emit_transcription(text, True)
                        transcription_found = True

                # Model turn (resposta do modelo em texto)
                if hasattr(server_content, 'model_turn') and server_content.model_turn:
                    model_turn = server_content.model_turn
                    if hasattr(model_turn, 'parts') and model_turn.parts:
                        for part in model_turn.parts:
                            if hasattr(part, 'text') and part.text:
                                text = part.text
                                print(f"🤖 Model response: {text}")
                                self._emit_transcription(text, True)
                                transcription_found = True

            # Método 2: Verificar diretamente na raiz da mensagem
            direct_attrs = ['input_transcription', 'output_transcription', 'transcription', 'text']
            for attr in direct_attrs:
                if hasattr(message, attr):
                    value = getattr(message, attr)
                    if value:
                        print(f"🎯 Encontrado {attr}: {value}")
                        if hasattr(value, 'text'):
                            text = value.text
                            print(f"📝 Transcription: {text}")
                            self._emit_transcription(text, True)
                            transcription_found = True
                        elif isinstance(value, str):
                            print(f"📝 Direct text: {value}")
                            self._emit_transcription(value, True)
                            transcription_found = True

            # Se não encontrou transcrição, fazer log completo para debug
            if not transcription_found:
                print(f"⚠️  Nenhuma transcrição encontrada na mensagem")
                if hasattr(message, '__dict__'):
                    # Fazer dump completo para entender a estrutura
                    try:
                        import json
                        message_dict = {}
                        for key, value in message.__dict__.items():
                            try:
                                message_dict[key] = str(value)[:500]  # Limitar tamanho
                            except:
                                message_dict[key] = f"<{type(value).__name__}>"
                        print(f"📋 Estrutura completa da mensagem: {json.dumps(message_dict, indent=2)}")
                    except Exception as dump_error:
                        print(f"❌ Erro ao fazer dump da mensagem: {dump_error}")

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
        """Envia áudio de forma assíncrona"""
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

                # Log do tamanho do chunk (só primeiros chunks para não poluir)
                if len(self.audio_buffer) < 5:
                    print(f"📤 Enviando áudio: {len(audio_bytes)} bytes (primeiro chunks)")

                # Sintaxe correta baseada na documentação oficial
                audio_part = Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/pcm;rate=16000"
                )
                await self.session.send(audio_part)

        except Exception as e:
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
                print(f"📋 Áudio adicionado ao buffer (total: {len(self.audio_buffer)})")
        else:
            if len(self.audio_buffer) <= 1:  # Evitar spam
                print("⚠️  Tentativa de envio com sessão inativa")
