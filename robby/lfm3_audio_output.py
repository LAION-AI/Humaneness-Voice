"""Convert a single audio-model response to the lab's observation format."""
import json
import re


def parse_audio_output(raw):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip()).strip()
    # Never present an unfinished private reasoning block as an observation.
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.S).strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        # Do not turn malformed JSON into a purported audio observation.
        if text.startswith(('{', '[')):
            try:
                from json_repair import loads
                value = loads(text)
            except (ImportError, ValueError, TypeError):
                return None
        else:
            return {'observations': [{'category': 'audio', 'description': text, 'relevance': None}]}
    if isinstance(value, list):
        value = {'observations': value}
    if not isinstance(value, dict):
        return None
    return value
