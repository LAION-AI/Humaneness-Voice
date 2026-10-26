"""Pinned public ASR alternatives; the existing TYGER encoder stays resident."""
import os
from pathlib import Path

MODELS={
    'nemotron':{'label':'Nemotron 3.5 · original','repo':'nvidia/nemotron-3.5-asr-streaming-0.6b',
        'revision':'bundled-with-tyger','kind':'native','mode':'Live Nemotron preview + neural speech-pause final check'},
    'moonshine_de':{'label':'Moonshine Streaming Small · German','repo':'moonshine-ai/moonshine-streaming-small-de',
        'revision':'56473ba32b22b484fdfc7eb03d34c393ff494197','kind':'moonshine',
        'files':['*.json','model.safetensors'],'mode':'Nemotron preview + German Moonshine final after a speech pause'},
    'nemotron_de':{'label':'Nemotron 3.5 · German telephony','repo':'BotgenossenGmbH/nemotron-3.5-asr-de-telephony',
        'revision':'afa1e25c8697ba55bc234087c46d954065cd7678','kind':'telephony',
        'files':['nemotron-3.5-asr-de-telephony.nemo','README.md','CHECKSUMS.txt'],
        'mode':'Nemotron preview + German telephony-adapted final after a speech pause'},
    'parakeet_v3':{'label':'Parakeet TDT v3 · multilingual','repo':'nvidia/parakeet-tdt-0.6b-v3',
        'revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b','kind':'parakeet',
        'files':['*.json','model.safetensors'],'mode':'Nemotron preview + Parakeet v3 final after a speech pause'},
    'whisper_turbo':{'label':'Whisper Large v3 Turbo · German','repo':'deepdml/faster-whisper-large-v3-turbo-ct2',
        'revision':'4df90f75321148c3a29a9e2351b7ddf8f5b115a8','kind':'whisper',
        'files':['*.json','model.bin'],'mode':'Hybrid: Nemotron preview / TYGER + CPU VAD + Whisper final'},
}

def cache_root():return Path(os.environ.get('ROBO_ASR_CACHE',str(Path.home()/'.cache/robby-asr')))

def default_model():
    key=os.environ.get('ROBO_ASR_MODEL','parakeet_v3')
    return key if key in MODELS else 'parakeet_v3'

def worker_url():
    from urllib.parse import urlparse
    value=os.environ.get('ROBO_ASR_MODELS_BASE','http://127.0.0.1:8801').rstrip('/')
    if urlparse(value).hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('ASR model worker must be local')
    return value
