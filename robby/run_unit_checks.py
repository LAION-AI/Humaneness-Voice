"""Offline unit contracts, with machine-readable results for the English report."""
import json,time,unittest
from pathlib import Path

MODULES=['test_audio_input','test_asr_priority','test_gpu_guard','test_enrollment_cues','test_recovery_rules','test_stop_commands','test_live_decisions','test_laya_policy','test_conversation_prompt','test_conversation_director','test_shared_turn','test_speaker_names','test_speaker_session','test_speaker_turns',
    'test_identity','test_speaker_teacher','test_bridge','test_performance',
    'speaker_v2.test_predictor','speaker_v4.test_v4','speaker_v5.test_v5','speaker_live.test_playback']

def main():
    MODULES.extend(['test_open_speakers','test_knowledge_tools','test_audio_transport','test_stop_feedback','test_stop_voice','test_release_scan','test_avatar_actions','test_avatar_commands','test_avatar_personality','test_bolt','test_pause_endpoint','test_asr_choice'])
    if 'test_visual_identity' not in MODULES:MODULES.append('test_visual_identity')
    suite=unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report={'finished':time.time(),'modules':MODULES,'tests':result.testsRun,'failures':len(result.failures),
        'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),
        'network_or_physical_microphone_used':False}
    path=Path(__file__).parent/'logs/unit-checks.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
