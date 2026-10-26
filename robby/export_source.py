"""Strict text-source-only Robo handoff; no configuration secrets or recordings."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    files=list(ROOT.glob('*.py'))
    files += [ROOT/name for name in ['README.md','VOICE_CAPTIONS.md','BRIDGE_PHRASES_PLAN.md','VOICE_ASSISTANT_TEST_PROTOCOL.md','voice_assistant_test_matrix.json','LICENSE','NOTICE','requirements.txt','requirements-asr.txt','config.example.json','.gitignore']]
    files.append(ROOT/'static/VOICE_ASSISTANT_TEST_PROTOCOL.md')
    for pattern in ['*.js','*.css','*.html','*.svg','LICENSE-MIT.txt']:files+=list((ROOT/'static').glob(pattern))
    files.append(ROOT/'static/emotion-catalog.json')
    for name in ('bolt-speed.json','bolt-speaker-speed.json','bolt-engine-comparison.json',
                 'bolt-socket-check.json','bolt-browser-check.json','bolt-browser-first-turn.json',
                 'bolt-tts-profile.json','bolt-precision.json','bolt-semantic-graph.json','bolt-local-compile.json',
                 'speech-pause-check.json','speech-pause-browser.json','long-replies-check.json','long-replies-browser.json',
                 'avatar-command-before.json','avatar-command-check.json','avatar-command-browser.json',
                 'asr-model-check.json','asr-model-browser.json','asr-controls-browser.json',
                 'openrouter-check.json','openrouter-browser.json','conversation-focus-check.json','conversation-focus-browser.json',
                 'rebalance-browser.json','rebalance-stop-browser.json','rebalance-permissions-browser.json','rebalance-decision-check.json',
                 'rebalance-before-group-fix.json','conversation-stress-check.json','conversation-stress-selma.json',
                 'conversation-continuity-check.json','conversation-continuity-before.json','conversation-continuity-browser.json',
                 'voice-test-fixtures.json','voice-assistant-test-results.json',
                 'adversarial-check.json','adversarial-harness-baseline.json','adversarial-before-fix.json',
                 'adversarial-after-window-fix.json','adversarial-after-level-fix.json',
                 'adversarial-fixtures.json','adversarial-plan.json','adversarial-input-audit.json',
                 'adversarial-tool-control.json','adversarial-tool-before-fix.json',
                 'adversarial-full-run.json','adversarial-targeted-run.json',
                 'adversarial-legacy-failure.json','adversarial-legacy-before-format-fix.json',
                 'completion-stop-before.json','completion-stop-decision.json','completion-stop-browser.json',
                 'internal-note-ui.json','internal-note-model.json','completion-stop-blocked.json','completion-stop-fixtures.json',
                 'voice-caption-timing.json','voice-caption-cpu-before.json','voice-caption-browser.json','voice-caption-model.json'):
        path=ROOT/'static'/name
        if path.exists():files.append(path)
    files+=list((ROOT/'ci').glob('*.yml'))
    for pattern in ['*.py','*.md']:files+=list((ROOT/'speaker_v2').glob(pattern))
    files+=list((ROOT/'speaker_v3').glob('*.py'))
    files+=list((ROOT/'speaker_v4').glob('*.py'))
    files+=list((ROOT/'speaker_v5').glob('*.py'))
    files+=list((ROOT/'speaker_live').glob('*.py'))
    files+=list((ROOT/'speed_study').glob('*.py'))
    calibration=ROOT/'speaker_live/calibration.json'
    if calibration.exists():files.append(calibration)
    design=ROOT/'design-inspection.json'
    if design.exists():files.append(design)
    manifest={};payload={}
    for path in sorted(set(files)):
        if path.is_symlink():raise ValueError('No symlinks in source export')
        raw=path.read_bytes();raw.decode('utf-8')
        if path.name=='bolt-precision.json':
            # Representative authored clips are embedded once in the report.
            # Keep the raw measurement record compact in the source release.
            record=json.loads(raw)
            for row in record.get('rows',[]):row.pop('listen_mp3_b64',None)
            raw=json.dumps(record,indent=2,ensure_ascii=False).encode()
        if path.name=='rebalance-before-group-fix.json' or path.name.startswith('adversarial-') and path.suffix=='.json':
            # Preserve every failed-run event; only remove JSON whitespace.
            raw=json.dumps(json.loads(raw),separators=(',',':'),ensure_ascii=False).encode()
        if re.search(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)',raw):
            raise ValueError('Potential credential in source export')
        name=path.relative_to(ROOT).as_posix();payload[name]=raw
        manifest[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    # Retain every synthetic stress attempt (including failures), compacting
    # JSON whitespace only. The new evidence adds about 8 MB of text.
    # The UTF-8, path allowlist and credential checks above still apply.
    if sum(len(raw) for raw in payload.values())>16_000_000:raise ValueError('Unexpected source export size')
    folder=ROOT/'releases';folder.mkdir(exist_ok=True)
    target=folder/'robby-source.zip';temporary=folder/'robby-source.tmp.zip'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in payload.items():archive.writestr(name,raw)
        archive.writestr('SOURCE_MANIFEST.json',json.dumps({'files':manifest,'private_recordings':0,'third_party_weights':0},indent=2))
    temporary.replace(target)
    print(json.dumps({'file':str(target),'files':len(payload),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
