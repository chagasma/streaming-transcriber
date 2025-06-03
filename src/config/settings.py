import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
    GOOGLE_CLOUD_LOCATION = os.getenv('GOOGLE_CLOUD_LOCATION')

    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000

    DEFAULT_LANGUAGE = 'pt-BR'

    DEFAULT_MODEL = 'nova-3-medical'
