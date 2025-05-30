from abc import ABC, abstractmethod


class TranscriptionStrategy(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def send_audio(self, data):
        pass
