"""Pinned upstream source paths; no changes to the shared TTS process."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parent
TYGER=Path(os.environ.get('ROBO_TYGER_SOURCE',str(ROOT/'vendor/tyger-s2s')))
sys.path.insert(0,str(TYGER))

TYGER_REVISION='97d011dcd4c8d2a117e54f00bf5b2d457cefa160'
BUNDLE_REVISION='a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6'
CACHE=Path(os.environ.get('ROBO_TYGER_CACHE',str(Path.home()/'.cache/robo-tyger/hub')))
BUNDLE=Path(os.environ.get('ROBO_TYGER_BUNDLE',str(CACHE/'models--nineninesix--tyger'/'snapshots'/BUNDLE_REVISION)))
