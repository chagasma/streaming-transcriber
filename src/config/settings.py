import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY')
    DEEPGRAM_API_KEY = os.getenv('DEEPGRAM_API_KEY')

    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000

    DEFAULT_LANGUAGE = 'pt-BR'
    DEFAULT_MODEL = 'nova-2'
