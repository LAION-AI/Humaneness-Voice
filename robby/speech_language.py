"""German robot replies by default; explicit language requests still win."""
import os,re

def reply_language(text,history=()):
    for phrase in [text,*[h.get('content','') for h in reversed(history) if h.get('role')=='user']]:
        if not isinstance(phrase,str):continue
        matches=list(re.finditer(r'\b(?:auf|in)\s+(deutsch|german|englisch|english)\b',phrase,re.I))
        for m in reversed(matches):
            if re.search(r'\b(?:nicht|not)\s*$',phrase[:m.start()],re.I):continue
            return 'German' if m[1].casefold() in ('deutsch','german') else 'English'
    return 'English' if os.environ.get('ROBO_REPLY_LANGUAGE','de')=='en' else 'German'
