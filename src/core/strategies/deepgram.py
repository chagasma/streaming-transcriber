from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from flask_socketio import SocketIO
from urllib.parse import quote

from src.config.settings import Config
from src.core.strategies.base import TranscriptionStrategy


class DeepgramStrategy(TranscriptionStrategy):
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio
        self.dg_connection = None
        self.recording = False
        self.deepgram = DeepgramClient(Config.DEEPGRAM_API_KEY)
        self.medical_keyterms = [
            "paracetamol", "ibuprofeno", "amoxicilina", "dipirona", "omeprazol",
            "losartana", "atenolol", "metformina", "glibenclamida", "salbutamol",
            "beclometasona", "cetirizina", "loratadina", "ranitidina", "captopril",
            "enalapril", "amlodipino", "sinvastatina", "atorvastatina", "clopidogrel",
            "diclofenaco", "nimesulida", "prednisona", "dexametasona", "fluoxetina",
            "sertralina", "amitriptilina", "diazepam", "clonazepam", "alprazolam",
            "carbamazepina", "fenitoína", "lamotrigina", "gabapentina", "pregabalina",
            "quetiapina", "risperidona", "olanzapina", "haloperidol", "metoclopramida",
            "domperidona", "ondansetrona", "furosemida", "espironolactona", "levotiroxina",
            "insulina", "gliclazida", "sitagliptina", "empagliflozina", "dapagliflozina",
            "cetoprofeno", "meloxicam", "piroxicam", "tenoxicam", "morfina",
            "tramadol", "fentanila", "cetoconazol", "fluconazol", "itraconazol",
            "quetiapina hemifumarato", "lercanidipino cloridrato", "pamidronato dissódico",
            "dexmedetomidina cloridrato", "anfotericina b lipossomal", "paclitaxel ligado albumina",
            "lisdexanfetamina dimesilato", "tirofibana cloridrato", "fingolimode cloridrato",
            "pazopanibe cloridrato", "bendamustina cloridrato", "palmitato de paliperidona",
            "nitroprusseto sódico diidratado", "mebutato de ingenol", "triptorrelina acetato",
            "perindorpil arginina", "sorafenibe tosilato", "fosaprepitanto dimeglumina",
            "esilato de nintedanibe", "lapatinibe ditosilato", "ziprasidona cloridrato monoidratado",
            "upadacitinibe hemi-hidratado", "cobimetinibe hemifumarato", "sacubitril valsartana sódica"
        ]

    def on_message(self, client, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        if len(sentence) == 0:
            return

        self.socketio.emit('transcription', {
            'text': sentence,
            'is_final': result.is_final,
            'model': Config.DEFAULT_MODEL
        })
        print(f"Medical Transcription: {sentence}")

    @staticmethod
    def on_error(client, error, **kwargs):
        print(f"Deepgram Error: {error}")

    def _build_keyterm_params(self):
        keyterm_params = []
        for term in self.medical_keyterms[:100]:  # ! docs só permite até 100
            keyterm_params.append(f"keyterm={quote(term)}")
        return "&".join(keyterm_params)

    def start(self):
        options = LiveOptions(
            model=Config.DEFAULT_MODEL,
            language=Config.DEFAULT_LANGUAGE,
            smart_format=True,
            interim_results=True,
            numerals=True,
            profanity_filter=False,  # Importante para termos médicos
            redact=False,  # Não reduzir termos médicos
        )

        self.dg_connection = self.deepgram.listen.websocket.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_message)
        self.dg_connection.on(LiveTranscriptionEvents.Error, self.on_error)

        keyterm_params = self._build_keyterm_params()
        if keyterm_params:
            print(f"Using {len(self.medical_keyterms[:100])} medical keyterms for enhanced recognition")

        if not self.dg_connection.start(options, addons={"keyterm": self.medical_keyterms[:100]}):
            raise Exception("Failed to connect to Deepgram Nova-3 Medical")

        self.recording = True

    def stop(self):
        if self.dg_connection:
            self.dg_connection.finish()
            self.dg_connection = None
        self.recording = False

    def send_audio(self, data):
        if self.recording and self.dg_connection:
            self.dg_connection.send(data)
