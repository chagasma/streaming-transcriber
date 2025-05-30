from strategies.transcription_strategy import TranscriptionStrategy


class TranscriptionContext:
    def __init__(self, strategy: TranscriptionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: TranscriptionStrategy):
        self.strategy = strategy

    def start(self):
        self.strategy.start()

    def stop(self):
        self.strategy.stop()

    def send_audio(self, data):
        self.strategy.send_audio(data)
