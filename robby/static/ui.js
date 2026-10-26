import {RoboClient} from './client.js';
import {createRobbyRig,controlsFromFace} from './rig.js';
import {profiles,pose,mix,ActingTimeline} from './emotions.js';
import {GazeTracker} from './gaze.js';
import {MultimodalUI} from './multimodal-ui.js';
import {organisePlayground} from './playground-layout.js';
import {AvatarActions} from './avatar-actions.js';
import {attachActionControls} from './avatar-action-controls.js';
import {attachPetting} from './avatar-petting.js';
import {ASRControls} from './asr-controls.js';
import {LLMControls} from './llm-controls.js';
import {LiveOverview} from './live-overview.js';
const $=id=>document.getElementById(id);
let ready=false,connected=false,mic=false,speaking=false,replyPending=false,partial=null,answer=null;
let sentenceMode=false;
let foregroundInputUntil=0;
let asrControls=null;
let llmControls=null;
const overview=new LiveOverview();
let petting=null,optionalAvatarReply=false,avatarCancelPending=false,lastPetSpeech=-Infinity,lastPetInvite=-Infinity,lastPlaySpeech=-Infinity,petInvites=0,invitedQuiet=false,lastIdleEvent=null,pendingPlayFallback=null;
const personalityStarted=performance.now();
let lastPersonalityUser=personalityStarted;
function cancelOptionalAvatar(){pendingPlayFallback=null;if(optionalAvatarReply&&!avatarCancelPending){avatarCancelPending=true;client.stopAudio();client.send('avatar:cancel');}}
function silentPlayFallback(){const steps=pendingPlayFallback;pendingPlayFallback=null;
 if(steps&&avatarActions.enabled&&avatarActions.idleEnabled&&!document.hidden&&!eyeContact?.state.active&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
  avatarActions.play(steps,'idle');avatarActions.lastIdleEnd=avatarActions.queue.at(-1)?.end??performance.now();avatarActions.nextIdle=avatarActions.lastIdleEnd+avatarActions.idleDelayMs;
 }}
let desiredGate=true,lastServerAudio=0,lastCapture=0,micStarted=0,serverAudioSeconds=0;
let desiredProactive=true;
let desiredHelpful=true;try{desiredHelpful=localStorage.getItem('robby.helpfulInterruptions')!=='false';}catch{}
let desiredOpenSpeakers=true;try{desiredOpenSpeakers=localStorage.getItem('robby.openSpeakers')!=='false';}catch{}
let desiredStopFeedback=false;try{desiredStopFeedback=localStorage.getItem('robby.stopFeedback')==='true';}catch{}
function savedThreshold(key,fallback,min,max){try{const raw=localStorage.getItem(key),v=raw===null?NaN:Number(raw);return Number.isFinite(v)&&v>=min&&v<=max?v:fallback;}catch{return fallback;}}
function saveThreshold(key,value){try{localStorage.setItem(key,String(value));}catch{/* Private-mode storage is optional. */}}
const DEFAULT_VOICE_THRESHOLD=.32,CAMERA_FPS=10,FACE_INTERVAL_MS=1000/CAMERA_FPS;
let desiredUtteranceThreshold=savedThreshold('robby.voiceThreshold',DEFAULT_VOICE_THRESHOLD,.3,.9),desiredFastThreshold=savedThreshold('robby.fastThreshold',null,.1,.95);
let desiredOpenVoiceThreshold=savedThreshold('robby.openVoiceThreshold',.38,.3,.9);
let desiredFocus=true;try{desiredFocus=localStorage.getItem('robby.conversationFocus')!=='false';}catch{}
let desiredFocusThreshold=savedThreshold('robby.focusThreshold',.28,.20,.65);
try{if(localStorage.getItem('robby.focusBalanceVersion')!=='2'){if(Math.abs(desiredFocusThreshold-.32)<.001)desiredFocusThreshold=.28;saveThreshold('robby.focusThreshold',desiredFocusThreshold);localStorage.setItem('robby.focusBalanceVersion','2');}}catch{}
let desiredSpeechPause=savedThreshold('robby.speechPauseMs',700,200,2000);
try{if(localStorage.getItem('robby.pauseBalanceVersion')!=='2'){if(desiredSpeechPause===400)desiredSpeechPause=700;saveThreshold('robby.speechPauseMs',desiredSpeechPause);localStorage.setItem('robby.pauseBalanceVersion','2');}}catch{}
$('speech-pause').value=desiredSpeechPause;$('speech-pause-value').value=desiredSpeechPause+' ms';
$('speech-pause').oninput=()=>{$('speech-pause-value').value=$('speech-pause').value+' ms';};
$('speech-pause').onchange=()=>{desiredSpeechPause=Number($('speech-pause').value);saveThreshold('robby.speechPauseMs',desiredSpeechPause);client.send('asr:pause_ms:'+desiredSpeechPause);};
function applySpeakerSettings(){client.send('identity:open:'+(desiredOpenSpeakers?'on':'off'));client.send('identity:utterance_threshold:'+(sentenceMode?desiredOpenVoiceThreshold:desiredUtteranceThreshold));if(sentenceMode){client.send('asr:pause_ms:'+desiredSpeechPause);client.send('identity:focus:enabled:'+(desiredFocus?'on':'off'));client.send('identity:focus:threshold:'+desiredFocusThreshold);}if(desiredFastThreshold!==null)client.send('identity:threshold:'+desiredFastThreshold);}
let cameraStream=null,landmarker=null,lastVideo=-1,face={},cameraBusy=false,cameraGeneration=0,multimodal,videoFrameAt=null,cameraStarting=null;
const gaze=new GazeTracker();let eyeContact=null;
const deskView=document.body.dataset.robotView==='desk3d';
let rig;
if(deskView){const {createDeskRobotRig}=await import('./desk-robot3d.js');rig=createDeskRobotRig($('avatar'));}
else{const avatar=await fetch('/static/robby.svg').then(r=>r.text());$('avatar').innerHTML=avatar;rig=createRobbyRig($('avatar').querySelector('svg'));}
const timeline=new ActingTimeline();
const avatarActions=new AvatarActions();attachActionControls(avatarActions);
for(const p of profiles){const o=document.createElement('option');o.value=p.slug;o.textContent=p.label;$('expression').append(o);}
window.robotFace={rig,timeline,profiles,pose,actions:avatarActions};
if(deskView){const {attachDeskControls}=await import('./desk-controls.js');attachDeskControls(rig);}
window.robotFace.gaze=gaze;
window.robotFace.cameraStats={targetFps:CAMERA_FPS,inferences:0};
if(deskView){const {attachEyeContactControls}=await import('./eye-contact-controls.js');eyeContact=attachEyeContactControls(rig,gaze,async()=>{if(!cameraStream)await $('camera').onclick();return !!cameraStream;});window.robotFace.eyeContact=eyeContact;}
const worklog=document.createElement('a');worklog.href='/static/worklog.html';worklog.target='_blank';worklog.rel='noopener';worklog.textContent='Open tasks, fixes and test log ↗';worklog.className='desk-caption';$('speaker-controls').after(worklog);
$('mirror').parentElement.lastChild.textContent=' Follow my position';
try{$('mirror').checked=localStorage.getItem('robby.followPosition')!=='false';$('camera-auto').checked=localStorage.getItem('robby.cameraAuto')!=='false';}catch{}
$('acting').onchange=()=>timeline.enabled=$('acting').checked;
$('robot-keypoints').onchange=()=>rig.showKeypoints($('robot-keypoints').checked);
function phase(text,kind=''){ $('phase-text').textContent=text;$('phase').className='phase '+kind; }
function bubble(text,role){$('chat').querySelector('.welcome')?.remove();const e=document.createElement('div');e.className='bubble '+role;e.textContent=text;$('chat').append(e);$('chat').scrollTop=$('chat').scrollHeight;return e;}
function showInteraction(m){
 $('allow-proactive').checked=m.enabled;
 if(typeof m.helpful_enabled==='boolean')$('allow-helpful').checked=m.helpful_enabled;
 $('interaction-rules').replaceChildren(...(m.rules||[]).map(r=>{const e=document.createElement('div');e.className='interaction-rule';e.textContent=r.condition+' → “'+r.reply+'” · '+(r.match_mode==='any_term'?'local keywords: '+(r.trigger_terms||[]).join(', '):'meaning-dependent rule · decision model');return e;}));
 const d=m.decision||m.last_decision;
 $('interaction-status').textContent=m.warning||(m.semantic_error?`Semantic decision unavailable (${m.semantic_error}). Direct commands and Stop reply remain available; very short voice matching is experimental.`:!m.enabled?'Proactive interruptions paused. You can still interrupt Robby.':
  d?.action==='interrupt_user'?`${d.kind==='helpful'?'Helpful interjection':'Agreed rule'} triggered · decision ${d.decision_ms} ms. Listening continues during the interjection.`:
  (m.rules||[]).length?`Live rules armed · checks up to every ${m.decision_interval_ms||320} ms · ${m.decision_model||'live controller'}.`:
  m.helpful_enabled?'Live usefulness decisions enabled. Agree a rule for an exact topic and spoken response.':'No agreed rules yet. Ask Robby to interrupt you under a specific condition.');
 const v=m.verification;
 $('live-interruption-health').textContent=!m.enabled?'Proactive interruptions OFF. You can still stop Robby.':!(m.rules||[]).length&&!m.helpful_enabled?'No agreed rules yet. Tell Robby when to interrupt you.':m.waiting_for?
  `${m.rules.length} rule(s) armed · ${m.waiting_for}${v?' · voice '+v.teacher+' / '+v.required+' · fast '+v.fast+' · long '+v.long:''}`:
  d?.action==='interrupt_user'?`Rule triggered · decision ${d.decision_ms} ms. Microphone remains active.`:'Agreed rules armed · listening continuously.';
}
const client=window.robo=new RoboClient({
 connection(on){connected=on;if(!on){llmControls?.disconnected();if($('conversation-focus-status'))$('conversation-focus-status').textContent='Waiting for the first conversation partner.';avatarActions.clear();petting?.reset();optionalAvatarReply=avatarCancelPending=false;replyPending=false;mic=false;gate.checked=desiredGate;identityModel.value='v4-long3';timeout.value=900;$('speaker-chips').replaceChildren();$('call').textContent='◉ Start talking';$('call').classList.remove('active');phase(ready?'Ready when you are':'Connecting…','ready');}},
 microphone(on){mic=on;micStarted=on?performance.now():0;lastCapture=lastServerAudio=0;serverAudioSeconds=0;$('mic-level').value=0;$('mic-status').textContent=on?'Microphone started. Waiting for audio frames…':'Microphone off. Click Start talking and allow microphone access.';$('call').textContent=on?'■ End conversation':'◉ Start talking';$('call').classList.toggle('active',on);phase(on?'Listening to you':'Ready when you are',on?'ready':'');},
 input(s){lastCapture=performance.now();if(s.rms>.003&&(!sentenceMode||performance.now()<foregroundInputUntil)){avatarActions.touch();cancelOptionalAvatar();} $('mic-level').value=Math.min(1,(s.rms||0)*12);$('mic-status').textContent=s.state==='ended'?'The microphone was disconnected. Start talking again.':`${s.rms>.001?'Audio detected':'Microphone active · quiet input'} · ${s.sent} packets sent · ${serverAudioSeconds.toFixed(1)} s received by server.`;},
 playing(on){speaking=on;phase(on?'Robby is speaking':mic?'Listening to you':'Ready when you are',on?'speaking':'ready');},
 audio(m){overview.audio(m);},
 sync(r){$('sync-status').textContent=`Audio sync: ${r.method} · output ${Math.round(r.output_ms)} ms · input ${r.input_ms===null?'unknown':Math.round(r.input_ms)+' ms'}${r.uncertain?' · device estimate incomplete':''}. No fixed playback offset.`;},
 message(m){
  asrControls?.message(m);
  llmControls?.message(m);overview.message(m);
  if(m.type==='speech_activity'&&m.foreground_speech)foregroundInputUntil=performance.now()+800;
  const foregroundMessage=m.type==='endpoint'||!sentenceMode||performance.now()<foregroundInputUntil;
  if(foregroundMessage&&['asr_preview','partial','endpoint'].includes(m.type)&& (m.text||m.transcript)?.trim()){lastPersonalityUser=performance.now();invitedQuiet=false;cancelOptionalAvatar();}
  if(m.type==='avatar_event'&&['petted','ask_pet','play_vocal'].includes(m.event)){
   if(m.accepted){optionalAvatarReply=true;replyPending=true;answer=null;}
   else if(optionalAvatarReply){optionalAvatarReply=avatarCancelPending=false;replyPending=false;silentPlayFallback();}
  }
  if(m.type==='reply')pendingPlayFallback=null;
  if(m.type==='turn_done')silentPlayFallback();
  if(['barge','stop_command','error','reply_cancelled'].includes(m.type))pendingPlayFallback=null;
  if(['turn_done','stop_command','error'].includes(m.type)){optionalAvatarReply=avatarCancelPending=false;}
  if(['endpoint','reply','backchannel'].includes(m.type)||(foregroundMessage&&['asr_preview','partial'].includes(m.type)))avatarActions.touch();
  if(['barge','stop_command','reply_cancelled'].includes(m.type)||m.type==='error'&&!avatarActions.queue.some(q=>q.source==='requested'))avatarActions.clear();
  if(m.type==='avatar_action')avatarActions.request(m.action_id,m.steps);
  if(m.type==='avatar_gesture'&&!speaking&&!replyPending)avatarActions.listen(m.kind);
  if(m.type==='reply'&&!avatarActions.hasRequested(m.avatar_action_id))avatarActions.set(m.reply_id,m.avatar_actions,m.performance,m.avatar_event?'optional':'reply');
  multimodal?.message(m);
  if(m.type==='ready'){sendRecentIdle();multimodal?.ready();client.send('interaction:gestures:'+(avatarActions.enabled&&avatarActions.listeningEnabled?'on':'off'));client.send('interaction:stop-feedback:'+(desiredStopFeedback?'on':'off'));}
  if(m.type==='ready'&&m.sentence_stream)setSentenceMode(true);
  if(m.type==='asr_settings'){$('speech-pause-status').textContent=`Wait for ${m.speech_pause_ms} ms without speech before sending · background sounds are ignored.`;}
  if(m.type==='speech_activity'){$('speech-pause-status').textContent=m.speech?'Speech detected · collecting your words.':m.pending?`Speech pause · ${Math.min(m.quiet_ms,2000)} / ${m.speech_pause_ms} ms`:'Listening · background sounds do not reset the speech timer.';}
  if(m.type==='stop_feedback_setting')$('stop-feedback').checked=!!m.enabled;
  if(m.type==='stop_feedback')$('asr-status').textContent='Stop accepted. A brief acknowledgment follows; you can interrupt it too.';
  if(m.type==='ready'){phase(mic?(desiredGate&&!desiredOpenSpeakers?'Say Hey Robbie to join':'Listening to you'):'Ready when you are','ready');if(m.identity_mode)identityModel.value=m.identity_mode;if(m.identity_enabled!==desiredGate)client.send('identity:'+(desiredGate?'on':'off'));applySpeakerSettings();gate.checked=desiredGate;$('asr-status').textContent=desiredGate?(desiredOpenSpeakers?'Speak clearly, then pause. New foreground voices can join without a wake phrase.':'Speaker filter ON. Say “Hey Robbie, ich bin …” to join. Recognition previews below are not sent to Luna.'):'Speaker filter OFF. Speak normally; TYGER submits your turn after a pause.';}
  if(m.type==='ready'&&m.interaction){showInteraction(m.interaction);if(m.interaction.enabled!==desiredProactive)client.send('interaction:'+(desiredProactive?'on':'off'));client.send('interaction:helpful:'+(desiredHelpful?'on':'off'));}
  if(m.type==='interaction')showInteraction(m);
  if(m.type==='input_status'){lastServerAudio=performance.now();serverAudioSeconds=m.seconds;}
  if(m.type==='asr_preview'&&m.text?.trim()&&(sentenceMode||(!speaking&&!replyPending))&&!(m.source==='speaker_candidate'&&m.status!=='enrolled'&&$('speaker-chips').childElementCount)){$('asr-preview').textContent=m.text;$('asr-status').textContent=sentenceMode?`Live transcript · waiting for a ${desiredSpeechPause} ms speech pause.`:m.status==='checking_identity'?'Speech recognized · checking speaker identity…':m.status==='enrolled'?'Voice enrolled · finish your sentence, then pause.':$('speaker-chips').childElementCount?'Speech recognition preview · not yet submitted.':desiredOpenSpeakers?'Checking a new voice · speak a clear sentence, then pause.':'Speech recognized (not submitted). Say “Hey Robbie, ich bin …”, or use Talk without speaker filter.';}
  if(m.type==='partial'){if(!partial)partial=bubble(m.text,'user partial');else partial.textContent=m.text;$('asr-preview').textContent=m.text;$('asr-status').textContent='Speech recognized · waiting for the end of your turn.';}
  if(m.type==='endpoint'){replyPending=true;const text=(m.speaker_name?m.speaker_name+': ':'')+m.transcript;if(partial){partial.classList.remove('partial');partial.textContent=text;partial=null;}else if(m.transcript)bubble(text,'user');answer=null;$('asr-preview').textContent=m.transcript;$('asr-status').textContent='Turn submitted'+(m.speaker_name?' · '+m.speaker_name:'')+'.';phase('Thinking of a reply');}
  if(m.type==='asr_sentence'){$('asr-preview').textContent=m.text;$('asr-status').textContent=m.queued?'Speech pause detected · your next turn is queued.':`Speech pause detected (${m.speech_pause_ms??desiredSpeechPause} ms) · sending your complete turn.`;}
  if(m.type==='pipeline_metrics'){$('local-pipeline-status').textContent=`${m.metrics.llm_model||'Language model'} · ${m.metrics.llm_actual_provider||m.metrics.llm_provider||'local'} · first token ${Math.round(m.metrics.llm_ttft_ms||0)} ms · first sentence ${Math.round(m.metrics.llm_first_sentence_ms||0)} ms · first audio ${Math.round(m.metrics.first_audio_ms||0)} ms · generation RTF ${m.metrics.generation_rtf??'—'}`+(m.metrics.response_complete===false?' · Response limit reached; ask Robby to continue.':'');}
  if(m.type==='reply'){timeline.set(m.reply_id,m.performance);if(!answer)answer=bubble(m.text,'assistant');else answer.textContent=m.text;$('subtitle').textContent=m.text.length>140?m.text.slice(0,137)+'…':m.text;
   if(m.sources?.length){const box=document.createElement('details');box.className='reply-sources';const title=document.createElement('summary');title.textContent='Sources consulted · '+m.sources.length;box.append(title);for(const s of m.sources){try{const url=new URL(s.url);if(url.protocol!=='https:')continue;const a=document.createElement('a');a.href=url.href;a.target='_blank';a.rel='noopener noreferrer';a.textContent=(s.publisher?s.publisher+' · ':'')+s.title;box.append(a,document.createElement('br'));}catch{}}answer.append(box);}
  }
  if(m.type==='state'){$('end-meter').value=m.p_end||0;$('barge-meter').value=m.yield||0;}
  if(m.type==='barge'){replyPending=false;timeline.clear();answer=null;phase('I’m listening','ready');$('subtitle').textContent='Go ahead. You can interrupt me naturally.';}
  if(m.type==='turn_done'){replyPending=false;answer=null;phase(mic?'Listening to you':'Ready when you are','ready');}
  if(m.type==='stop_command'){replyPending=false;answer=null;timeline.clear();client.stopAudio();phase('Paused · I’m listening','ready');$('subtitle').textContent='Stopped. Say what you would like next.';$('asr-status').textContent=m.identity_uncertain?'Foreground Stop accepted. The short voice cue was uncertain; Robby is listening.':'Stop request accepted from '+(m.speaker_name||'the current speaker')+'. No new spoken reply.';}
  if(m.type==='error'){replyPending=false;const text=m.error==='audio_too_fast'?'The voice server fell behind the microphone. End the conversation and reconnect; if it happens again, the shared GPU service needs checking.':m.error;phase('Something needs attention');$('subtitle').textContent=text;$('asr-status').textContent='Audio processing error: '+text;}
  if(m.type==='ignored'&&!replyPending&&!speaking){phase(m.source==='visual_filter'?'Camera could not confirm speech':'Waiting for an enrolled speaker');$('subtitle').textContent=m.reason;$('asr-status').textContent=m.reason;partial=null;}
  if(m.type==='identity'){
   if(typeof m.open_enrollment==='boolean')openSpeakers.checked=m.open_enrollment;
   if(typeof m.enabled==='boolean'){gate.checked=m.enabled;desiredGate=m.enabled;}
   if(m.mode)identityModel.value=m.mode;
   const p=m.decision||m.prediction||{};
   if(typeof m.threshold==='number'){strictness.value=m.threshold;strictValue.value=m.threshold.toFixed(2);}
   if(typeof m.utterance_threshold==='number'){utteranceStrictness.value=m.utterance_threshold;$('utterance-threshold-value').value=m.utterance_threshold.toFixed(2);}
   if(m.verification){const v=m.verification,scores=[...(v.reference_scores||[])].sort((a,b)=>b.cosine-a.cosine),best=scores[0];if(best){const gap=scores.length>1?' · gap '+(best.cosine-scores[1].cosine).toFixed(2):'';matchFeedback.textContent=`Last sentence: ${best.name} · voice match ${best.cosine.toFixed(2)} / required ${Number(v.threshold).toFixed(2)}${gap}${v.view_cosine==null?'':' · view agreement '+v.view_cosine.toFixed(2)}. Cosine is not a probability.`;}}
   $('speaker-chips').replaceChildren(...(m.speakers||[]).map(s=>{const e=document.createElement('span');e.className='speaker-chip';e.textContent=s.name+(s.admitted===false?' · outside conversation':'')+(s.reference_ready?'':' · joining…')+(s.voice_variants>1?' · '+s.voice_variants+' voice variants':'');return e;}));
   $('identity-status').textContent=`Speaker cue: ${p.name||p.kind||'waiting'}${p.similarity==null?'':' · cosine '+p.similarity.toFixed(2)}. Enrolled: ${(m.users||[]).join(', ')||'none — say Hey Robbie, I’m …'}. ${m.enabled?'Filter ON · '+(m.diarization_active?'Nemotron diarization + V4 · ':'embedding-only · ')+'submit after ~1 s without this speaker.':'Filter off: original TYGER endpointing.'} RAM only; not authentication.`;
  }
  if(m.type==='warning'){$('identity-status').textContent=m.text;$('asr-status').textContent=m.text;}
  if(m.type==='identity'&&m.sentence_mode){setSentenceMode(true);const f=m.conversation_focus;focusStatus.textContent=m.handover_pending?'Ready for the next person. Speak a complete sentence now; no introduction is required.':f?.enabled?`Conversation partner: ${f.name||'waiting for the first voice'} · match threshold ${Number(f.threshold).toFixed(2)} · focus expires after 3 minutes of inactivity.`:'Focus disabled: all intelligible voices can join and interrupt.';$('identity-status').textContent=`Session speakers: ${(m.speakers||[]).map(s=>s.name+' · '+s.voice_variants+' voice reference(s)').join(', ')||'none yet'}. Names are optional; say “I am also Christoph” to link a voice variant.`;}
  if(m.type==='speaker_focus'){focusStatus.textContent=(m.accepted?'Current partner confirmed':'Background or uncertain voice ignored')+(m.similarity==null?'':` · voice cosine ${m.similarity.toFixed(2)}`)+'.';}
  if(m.type==='speaker_permission'){focusStatus.textContent=m.action==='invite'?(m.waiting_for_next_voice?`${m.name||'Your guest'} can speak next. The current person remains in the conversation.`:`${m.name||'Your guest'} is included in this conversation.`):'Background voices stay outside this conversation. Robby will not keep asking about them.';}
  if(m.type==='busy'){$('subtitle').textContent=m.text;}
  if(m.type==='service_status'){if(m.stage==='interjection_ready'){$('live-interruption-health').textContent=m.text;return;}phase(m.stage==='planning'?'Thinking of a useful reply':m.stage==='preparing'?'Preparing a spoken reply':'Waiting for the shared voice model');$('subtitle').textContent=m.text;$('asr-status').textContent=m.text;}
  if(m.type==='calibration'){
   if(m.available){client.clock.acousticLatency=m.delay_ms/1000;$('sync-status').textContent=`Measured extra echo delay: ${m.delay_ms.toFixed(1)} ms · correlation ${m.confidence.toFixed(2)}.`;}
   else $('sync-status').textContent='Echo calibration uncertain: '+m.reason+'. Keeping device-timestamp mapping; no extra delay guessed.';
  }
 }
});
asrControls=new ASRControls(client);window.robotFace.asrControls=asrControls;
llmControls=new LLMControls(client);window.robotFace.llmControls=llmControls;window.robotFace.overview=overview;
multimodal=new MultimodalUI(client,$('speaker-controls'));window.robotFace.multimodal=multimodal;
const canPersonalitySpeak=()=>connected&&ready&&!speaking&&!replyPending&&!$('message').value.trim()&&!document.hidden&&!eyeContact?.state.active;
const avatarEvent=event=>{if(['petted','ask_pet','play_vocal'].includes(event.event)){optionalAvatarReply=true;avatarCancelPending=false;replyPending=true;}client.send('avatar:'+JSON.stringify({...event,play_intensity:avatarActions.playIntensity,cuddle_intensity:avatarActions.cuddleIntensity}));};
function sendRecentIdle(){if(lastIdleEvent&&performance.now()-lastIdleEvent.at<600000)avatarEvent({event:'idle_play',...lastIdleEvent.event,age_ms:Math.round(performance.now()-lastIdleEvent.at)});}
avatarActions.onIdle=event=>{
 lastIdleEvent={event,at:performance.now()};
 const now=performance.now();
 if(event.carousel&&avatarActions.playVoiceEnabled&&canPersonalitySpeak()&&now-lastPlaySpeech>=45000){
  lastPlaySpeech=now;pendingPlayFallback=[{kind:'carousel',direction:'clockwise',duration_s:6},{kind:'afterglow',duration_s:6}];
  avatarEvent({event:'play_vocal',...event});
 }else if(connected)avatarEvent({event:'idle_play',...event});
};
// Invitations remain independent of the silliness slider and never compete
// with conversation. Server-side limits independently enforce these bounds.
setInterval(()=>{const now=performance.now();
 if(!canPersonalitySpeak()||!avatarActions.enabled||!avatarActions.petInvitesEnabled||avatarActions.cuddleIntensity<=0||invitedQuiet||petInvites>=2||
    now-personalityStarted<avatarActions.inviteStartMs||now-lastPetInvite<avatarActions.inviteCooldownMs||now-lastPersonalityUser<60000||petting?.stroke.active)return;
 invitedQuiet=true;petInvites++;lastPetInvite=now;avatarEvent({event:'ask_pet'});
},2000);
petting=attachPetting({element:$('avatar'),rig,actions:avatarActions,allowed:()=>!eyeContact?.state.active,
 onStart:()=>{lastPersonalityUser=performance.now();invitedQuiet=false;if(ready)void client.start(false).catch(()=>{});},
 onPet:event=>{if(canPersonalitySpeak()&&avatarActions.petSpeechEnabled&&avatarActions.cuddleIntensity>0&&performance.now()-lastPetSpeech>=avatarActions.petCooldownMs){
  lastPetSpeech=performance.now();avatarEvent({event:'petted',duration_ms:event.duration_ms});return true;
 }return false;} });window.robotFace.petting=petting;
for(const id of ['avatar-actions-enabled','avatar-idle-enabled','avatar-play-voice-enabled','avatar-petting-enabled','avatar-pet-speech-enabled','avatar-pet-invites-enabled'])$(id).addEventListener('change',()=>{petting.reset();cancelOptionalAvatar();});
for(const id of ['avatar-play-intensity','avatar-cuddle-intensity'])$(id).addEventListener('input',()=>{petting.reset();cancelOptionalAvatar();});
document.addEventListener('visibilitychange',()=>{if(document.hidden)cancelOptionalAvatar();});
for(const id of ['avatar-actions-enabled','avatar-listening-enabled'])$(id).addEventListener('change',()=>client.send('interaction:gestures:'+(avatarActions.enabled&&avatarActions.listeningEnabled?'on':'off')));
$('call').onclick=async()=>{try{if(mic)client.stop();else await client.start(true);}catch(e){$('subtitle').textContent=e.message;phase('Microphone unavailable — try typing');}};
$('unfiltered-talk').onclick=async()=>{desiredGate=false;try{await client.start(true);client.send('identity:off');if(sentenceMode){desiredFocus=false;focusToggle.checked=false;saveThreshold('robby.conversationFocus',false);client.send('identity:focus:enabled:off');}gate.checked=false;$('asr-status').textContent='Speaker filter OFF. Speak normally; background voices may also be transcribed.';phase('Listening to you','ready');}catch(e){$('mic-status').textContent='Microphone unavailable: '+e.message;}};
setInterval(()=>{if(!mic)return;const now=performance.now();if(now-micStarted<2500)return;
 if(!lastCapture||now-lastCapture>2500)$('mic-status').textContent=`No audio frames from the microphone (${client.ctx?.state||'no audio context'}). Check browser permission/input device, then End conversation and Start talking.`;
 else if(!lastServerAudio||now-lastServerAudio>4000)$('mic-status').textContent='Microphone captures audio, but the server has not acknowledged recent packets. Reconnect the conversation.';
},1000);
$('interrupt').onclick=()=>{avatarActions.clear();timeline.clear();client.stopAudio();client.send('interrupt');};
$('allow-proactive').onchange=()=>{desiredProactive=$('allow-proactive').checked;client.send('interaction:'+(desiredProactive?'on':'off'));if(!connected)$('interaction-status').textContent=desiredProactive?'Enabled for your next conversation. Agree a rule with Robby first.':'Paused for your next conversation. You can still interrupt Robby.';};
$('allow-helpful').checked=desiredHelpful;$('allow-helpful').onchange=()=>{desiredHelpful=$('allow-helpful').checked;try{localStorage.setItem('robby.helpfulInterruptions',String(desiredHelpful));}catch{}client.send('interaction:helpful:'+(desiredHelpful?'on':'off'));};
$('clear-interaction-rules').onclick=()=>{client.send('interaction:clear');if(!connected){$('interaction-rules').replaceChildren();$('interaction-status').textContent='Rules are session-only. No active conversation.';}};
const calibrate=document.createElement('button');calibrate.className='text-button';calibrate.textContent='Calibrate acoustic echo';calibrate.onclick=async()=>{try{await client.start(true);client.clock.acousticLatency=0;client.send('calibrate');$('sync-status').textContent='Listening for a short, quiet calibration tone…';}catch(e){$('sync-status').textContent=e.message;}};$('sync-status').after(calibrate);
const stopLabel=document.createElement('label'),stopFeedback=document.createElement('input');stopFeedback.type='checkbox';stopFeedback.id='stop-feedback';stopFeedback.checked=desiredStopFeedback;stopLabel.append(stopFeedback,' Brief spoken acknowledgment after I stop Robby (otherwise stay silent)');$('allow-helpful').parentElement.after(stopLabel);
const stopHelp=document.createElement('p');stopHelp.textContent='Keep your microphone on. Say “Stopp”, “Hör bitte auf” or “Warte bitte” while Robby speaks. The current speaker is checked continuously, independently of proactive-interruption settings. Wait/quiet commands always stay silent. Acknowledgments yield to continuing speech.';stopLabel.after(stopHelp);
stopFeedback.onchange=()=>{desiredStopFeedback=stopFeedback.checked;saveThreshold('robby.stopFeedback',desiredStopFeedback);client.send('interaction:stop-feedback:'+(desiredStopFeedback?'on':'off'));};
async function submit(text){if(!ready)return;lastPersonalityUser=performance.now();avatarActions.touch();invitedQuiet=false;try{
 overview.typed(text);
 if(optionalAvatarReply){cancelOptionalAvatar();for(let i=0;i<40&&optionalAvatarReply;i++)await new Promise(r=>setTimeout(r,50));}
 await client.start(false);sendRecentIdle();bubble(text,'user');answer=null;replyPending=true;client.send('text:'+text);phase('Thinking of a reply');}catch(e){replyPending=false;$('subtitle').textContent=e.message;}}
$('message').addEventListener('input',()=>{lastPersonalityUser=performance.now();avatarActions.touch();cancelOptionalAvatar();});
$('form').onsubmit=e=>{e.preventDefault();const text=$('message').value.trim();if(text){$('message').value='';submit(text);}};
for(const b of document.querySelectorAll('[data-text]'))b.onclick=()=>submit(b.dataset.text);
$('clear').onclick=()=>{timeline.clear();client.stop();$('chat').replaceChildren();overview.clear();partial=answer=null;};
const gateLabel=document.createElement('label'),gate=document.createElement('input');gate.type='checkbox';gate.id='identity-gate';gate.disabled=true;gate.checked=true;gateLabel.append(gate,' Recognize enrolled speakers · ignore background turns');$('speaker-controls').append(gateLabel);
const openLabel=document.createElement('label'),openSpeakers=document.createElement('input');openSpeakers.type='checkbox';openSpeakers.id='open-speakers';openSpeakers.checked=desiredOpenSpeakers;openLabel.append(openSpeakers,' Welcome new voices without “Hey Robbie”');gateLabel.after(openLabel);
const openHelp=document.createElement('p');openHelp.textContent='ON: clear new foreground voices can join as temporary guests; Robby asks their name once. Say “It’s me, Bob” / “Ich bin’s doch, Bob” in a clear sentence to explicitly add a voice variant to an existing name (up to four). This trusts your statement, not secure authentication. Background speech can also be admitted: switch OFF for wake-phrase-only enrollment. All voiceprints remain in session RAM.';openLabel.after(openHelp);
openSpeakers.onchange=()=>{desiredOpenSpeakers=openSpeakers.checked;saveThreshold('robby.openSpeakers',desiredOpenSpeakers);client.send('identity:open:'+(desiredOpenSpeakers?'on':'off'));};
const modelLabel=document.createElement('label');modelLabel.textContent='Speaker recognition ';const identityModel=document.createElement('select');identityModel.id='identity-model';for(const [value,label] of [['v4-long3','V4 · fast 320 ms + longer 3.04-s identity context (default)'],['v4-long','V4 · fast 320 ms + 2-s identity context'],['v4','V4 · 320-ms layer-mix only'],['ecapa','ECAPA comparison · legacy endpoint gate']]){const o=document.createElement('option');o.value=value;o.textContent=label;o.disabled=value==='v4-long3';identityModel.append(o);}identityModel.value='v4-long3';modelLabel.append(identityModel);gateLabel.after(modelLabel);identityModel.onchange=async()=>{const desired=identityModel.value;try{await client.start(true);client.send('identity:mode:'+desired);applySpeakerSettings();gate.checked=desired.startsWith('v4');gate.disabled=false;}catch(e){$('identity-status').textContent=e.message;}};
const localPipeline=document.createElement('p');localPipeline.id='local-pipeline-status';localPipeline.hidden=true;$('asr-status').after(localPipeline);
const focusBox=document.createElement('section');focusBox.id='conversation-focus-controls';focusBox.hidden=true;
const focusLabel=document.createElement('label'),focusToggle=document.createElement('input');focusToggle.type='checkbox';focusToggle.id='conversation-focus';focusToggle.checked=desiredFocus;focusLabel.append(focusToggle,' Stay with the current conversation partner');
const focusHelp=document.createElement('p');focusHelp.textContent='Say “This is my friend Alice, talk to her too” or “Das ist mein Kumpel Bob, rede auch mit ihm” to include a guest. Both people can then keep talking. Say “Ignore the background voices” to stop questions about them, or “Talk only to me” to narrow the conversation. Voice changes and explicit name corrections can share one session profile.';
const focusThresholdLabel=document.createElement('label');focusThresholdLabel.textContent='Partner match · lower tolerates more voice variation ';
const focusRange=document.createElement('input'),focusValue=document.createElement('output');focusRange.id='conversation-focus-threshold';focusRange.type='range';focusRange.min=.20;focusRange.max=.65;focusRange.step=.01;focusRange.value=desiredFocusThreshold;focusValue.value=desiredFocusThreshold.toFixed(2);focusThresholdLabel.append(focusValue,focusRange);
const releaseFocus=document.createElement('button');releaseFocus.type='button';releaseFocus.id='release-conversation-focus';releaseFocus.className='secondary';releaseFocus.textContent='Let another person join';
const focusStatus=document.createElement('p');focusStatus.id='conversation-focus-status';focusStatus.setAttribute('role','status');focusStatus.textContent='Waiting for the first conversation partner.';
const focusSettings=document.createElement('details'),focusSummary=document.createElement('summary');focusSummary.textContent='Voice tolerance & switching people';focusSettings.append(focusSummary,focusHelp,focusThresholdLabel,releaseFocus);
const focusReport=document.createElement('a');focusReport.href='/static/rebalance-update.html';focusReport.target='_blank';focusReport.rel='noopener';focusReport.textContent='Tests, thresholds and known limits ↗';focusSettings.append(focusReport);
focusBox.append(focusLabel,focusStatus,focusSettings);$('quick-controls').append(focusBox);
focusToggle.onchange=()=>{desiredFocus=focusToggle.checked;saveThreshold('robby.conversationFocus',desiredFocus);client.send('identity:focus:enabled:'+(desiredFocus?'on':'off'));};
focusRange.oninput=()=>focusValue.value=Number(focusRange.value).toFixed(2);
focusRange.onchange=()=>{desiredFocusThreshold=Number(focusRange.value);saveThreshold('robby.focusThreshold',desiredFocusThreshold);client.send('identity:focus:threshold:'+desiredFocusThreshold);};
releaseFocus.onclick=async()=>{try{await client.start(false);client.stopAudio();client.send('identity:focus:release');focusStatus.textContent='Ready for the next person. Speak a complete sentence now; no wake phrase is required.';}catch(e){focusStatus.textContent=e.message;}};
function setSentenceMode(on){sentenceMode=on;if(!on)return;
 focusBox.hidden=false;
 $('speech-pause-controls').hidden=false;
 if(!identityModel.querySelector('[value="open-sentences"]'))identityModel.add(new Option('ECAPA voice profiles · open sentence streaming','open-sentences'));identityModel.value='open-sentences';
 gate.checked=false;gate.disabled=true;gateLabel.lastChild.textContent=' Voice labels use full-utterance ECAPA; conversation focus is controlled above';
 openSpeakers.checked=true;openSpeakers.disabled=true;identityModel.disabled=true;modelLabel.title='Open sentence mode uses full-utterance ECAPA reference embeddings.';
 openHelp.hidden=true;
 $('speaker-controls').querySelector('strong').textContent='Keep the conversation together';
 gateLabel.title='The independent conversation-focus switch guards background speech in this mode.';
 localPipeline.hidden=false;
 const pill=document.querySelector('.brand-note');if(pill)pill.textContent='sentence-streamed voice chat';
 matchHelp.textContent='Voice-label threshold: 0.38 by default. During an active conversation the separate partner threshold (default 0.28) preserves identity across prosody changes. Say “I am also Christoph” to link a confirmed voice variant.';
 utteranceStrictness.value=desiredOpenVoiceThreshold;utteranceValue.value=desiredOpenVoiceThreshold.toFixed(2);
}
gate.onchange=async()=>{const enabled=gate.checked;desiredGate=enabled;try{await client.start(true);client.send('identity:'+(enabled?'on':'off'));$('asr-status').textContent=enabled?'Speaker filter ON. Say “Hey Robbie, ich bin …” to join.':'Speaker filter OFF. Speak normally; background voices may also be transcribed.';}catch(e){gate.checked=false;desiredGate=false;$('identity-status').textContent=e.message;}};
const timeoutLabel=document.createElement('label');timeoutLabel.textContent='Forget inactive users after ';const timeout=document.createElement('select');for(const [value,label] of [[60,'1 minute'],[300,'5 minutes'],[900,'15 minutes'],[0,'this conversation']]){const o=document.createElement('option');o.value=value;o.textContent=label;timeout.append(o);}timeout.value=900;timeout.onchange=()=>client.send('identity:timeout:'+timeout.value);timeoutLabel.append(timeout);modelLabel.after(timeoutLabel);
const tuning=document.createElement('section');tuning.className='speaker-tuning';tuning.setAttribute('aria-label','Voice matching sensitivity');gateLabel.before(tuning);
const utteranceLabel=document.createElement('label');utteranceLabel.textContent='Voice match strictness · lower accepts more ';const utteranceStrictness=document.createElement('input');utteranceStrictness.id='utterance-strictness';utteranceStrictness.type='range';utteranceStrictness.min=.3;utteranceStrictness.max=.9;utteranceStrictness.step=.01;utteranceStrictness.value=desiredUtteranceThreshold;const utteranceValue=document.createElement('output');utteranceValue.id='utterance-threshold-value';utteranceValue.value=desiredUtteranceThreshold.toFixed(2);utteranceLabel.append(utteranceValue,utteranceStrictness);tuning.append(utteranceLabel);
const matchHelp=document.createElement('p');matchHelp.textContent='Default: 0.32 cosine. Lower values accept more, including possible background voices; raise this if the wrong person is accepted. Changes apply immediately and are remembered in this browser. Reset restores 0.32; enrolled voiceprints are not saved.';tuning.append(matchHelp);
const matchFeedback=document.createElement('p');matchFeedback.id='speaker-match-feedback';matchFeedback.setAttribute('role','status');matchFeedback.textContent='Last sentence voice match: waiting for speech.';tuning.append(matchFeedback);
const resetThresholds=document.createElement('button');resetThresholds.className='text-button';resetThresholds.textContent='Reset voice matching';tuning.append(resetThresholds);
utteranceStrictness.oninput=()=>utteranceValue.value=(+utteranceStrictness.value).toFixed(2);utteranceStrictness.onchange=()=>{if(sentenceMode){desiredOpenVoiceThreshold=+utteranceStrictness.value;saveThreshold('robby.openVoiceThreshold',desiredOpenVoiceThreshold);client.send('identity:utterance_threshold:'+desiredOpenVoiceThreshold);}else{desiredUtteranceThreshold=+utteranceStrictness.value;saveThreshold('robby.voiceThreshold',desiredUtteranceThreshold);client.send('identity:utterance_threshold:'+desiredUtteranceThreshold);}};
const advanced=document.createElement('details');advanced.className='speaker-advanced';advanced.innerHTML='<summary>Advanced · fast speaker cue</summary>';timeoutLabel.after(advanced);
const strictLabel=document.createElement('label');strictLabel.textContent='Fast cue strictness ';const strictness=document.createElement('input');strictness.id='speaker-strictness';strictness.type='range';strictness.min=.1;strictness.max=.95;strictness.step=.01;strictness.value=desiredFastThreshold??.62;const strictValue=document.createElement('output');strictValue.value=Number(strictness.value).toFixed(2);strictness.oninput=()=>strictValue.value=(+strictness.value).toFixed(2);strictness.onchange=()=>{desiredFastThreshold=+strictness.value;saveThreshold('robby.fastThreshold',desiredFastThreshold);client.send('identity:threshold:'+desiredFastThreshold);};strictLabel.append(strictValue,strictness);advanced.append(strictLabel);
const advancedHelp=document.createElement('p');advancedHelp.textContent='Only lower this if the fast cue keeps losing you. Full-sentence ECAPA, overlap, word-origin and playback-echo checks remain active. Voice interruptions additionally verify the current addressee against other users and Robby; short stop commands need both fast and teacher evidence.';advanced.append(advancedHelp);
resetThresholds.onclick=()=>{desiredUtteranceThreshold=DEFAULT_VOICE_THRESHOLD;desiredFastThreshold=null;try{localStorage.removeItem('robby.voiceThreshold');localStorage.removeItem('robby.fastThreshold');}catch{}utteranceStrictness.value=DEFAULT_VOICE_THRESHOLD;utteranceValue.value=DEFAULT_VOICE_THRESHOLD.toFixed(2);strictness.value=.62;strictValue.value='0.62';applySpeakerSettings();client.send('identity:threshold:0.619811549782753');};
const forget=document.createElement('button');forget.className='text-button';forget.textContent='Forget enrolled users';forget.onclick=()=>client.send('identity:clear');timeoutLabel.after(forget);
const studyLink=document.createElement('a');studyLink.href='/speaker-study/';studyLink.target='_blank';studyLink.rel='noopener';studyLink.textContent='Speaker predictor study ↗';$('identity-status').after(studyLink);
const implementationLink=document.createElement('a');implementationLink.href='/static/report.html';implementationLink.target='_blank';implementationLink.rel='noopener';implementationLink.textContent='Implementation, tests and limitations ↗';studyLink.after(document.createElement('br'),implementationLink);
const latencyReport=document.createElement('a');latencyReport.href='/static/robby-local-speed-report.html';latencyReport.target='_blank';latencyReport.rel='noopener';latencyReport.textContent='Local LLM, TTS precision and latency measurements ↗';implementationLink.after(document.createElement('br'),latencyReport);
const identityHelp=document.createElement('p');identityHelp.textContent='Default: streaming diarization, fast 320-ms speaker cues and a longer 3.04-s identity context. This longer context uses speech already heard; it does not extend the one-second pause. Transcript and speaker checks are prepared during that pause and discarded if you continue speaking. Names and voice memory are session-only; uncertain overlaps can require repetition. The 2-s option remains available for comparison.';studyLink.after(identityHelp);
async function poll(){try{const s=await fetch('/api/state').then(r=>r.json());ready=s.ready;gate.disabled=!s.identity_available;$('call').disabled=!ready;const long=s.identity_models?.long,long3=s.identity_models?.long3;identityModel.querySelector('[value="v4-long3"]').disabled=!long3;if(long)identityModel.querySelector('[value="v4-long"]').textContent='V4 320 ms + '+long.label;if(long3)identityModel.querySelector('[value="v4-long3"]').textContent='V4 320 ms + '+long3.label;if(!connected){phase(s.ready?'Ready when you are':s.error||s.phase,s.ready?'ready':'');$('identity-status').textContent='Speaker extension: '+s.identity_model+'. Voiceprints stay in RAM for this conversation only.';}if(s.sentence_stream){setSentenceMode(true);identityHelp.textContent='ASR collects all sentences until your selected speech pause. A small CPU neural detector separates speech from background sounds. Output sentences still stream immediately to TTS. Speech after a submitted turn enters the next request; an LLM request already running cannot be edited in place.';}}catch{phase('Reconnecting…');}}
poll();setInterval(poll,10000);

async function startCamera(){
 if(cameraStream)return true;if(cameraStarting)return cameraStarting;
 cameraStarting=openCamera();try{return await cameraStarting;}finally{cameraStarting=null;$('camera').disabled=false;}
}
function stopCamera(){
 if(cameraStream){if(eyeContact?.state.active)eyeContact.finish();cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;}
 cameraGeneration++;cameraBusy=false;multimodal.setCamera('off');gaze.resetTracking();$('video').hidden=true;$('mesh').hidden=true;$('camera').textContent='Enable camera';$('camera-status').textContent='Camera off; audio-only fallback.';$('follow-status').textContent='Camera off. Click Enable camera to follow your position.';face={};
}
$('camera').onclick=async()=>{
 if(cameraStream){stopCamera();$('camera-auto').checked=false;try{localStorage.setItem('robby.cameraAuto','false');}catch{}return;}
 return startCamera();
};
async function openCamera(){
 try{
  $('camera').disabled=true;$('camera').textContent='Starting camera…';
  $('camera-status').textContent='Loading local face tracking…';
  if(!landmarker){
   landmarker=new Worker('/static/face-camera-worker.js');
   await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Camera worker load timed out')),30000);
    landmarker.onerror=e=>{clearTimeout(timer);reject(Error('Camera worker unavailable: '+e.message));};
    landmarker.onmessage=e=>{if(e.data.type==='ready'){clearTimeout(timer);resolve();}else if(e.data.type==='error'){clearTimeout(timer);reject(Error(e.data.error));}};landmarker.postMessage({type:'load'});
   });
   landmarker.onmessage=({data:d})=>{cameraBusy=false;if(d.type==='error'){$('camera-status').textContent='Camera inference unavailable: '+d.error;multimodal.setCamera('unavailable');return;}
    if(d.type!=='result'||d.generation!==cameraGeneration||!cameraStream)return;
    const video=$('video');let selected;
    try{selected=multimodal.frame(d.result,d.at,video,d.ms);}
    catch(e){window.robotFace.cameraStats.recognitionError=e.message;selected=d.result.faceLandmarks?.length===1?d.result:null;}
    window.robotFace.cameraStats.inferences++;
    window.robotFace.cameraStats.lastMs=d.ms;
    if(selected){face=controlsFromFace(selected);gaze.observe(selected,video.videoWidth,video.videoHeight,d.at);}else face={};
    const target=gaze.read();
    $('follow-status').textContent=eyeContact?.state.frozen?'Calibration frozen · press Y to save, Esc to resume, or Resume eye contact.':!$('mirror').checked?'Position following paused.':rig.state?.manual||rig.state?.tour?'Manual pose overrides the camera · click Resume eye contact.':target?`Following ${multimodal.followedTrack||'your eyes'} · pan ${target.pan.toFixed(0)}° / tilt ${target.tilt.toFixed(0)}° · position, not rotation.`:'Camera on · looking for fresh eye landmarks.';
    $('camera-status').textContent=`Local CPU worker · ${d.result.faceLandmarks?.length||0} faces · ${Math.round(d.ms)} ms · up to 10 FPS. No camera images leave this browser.`;
    const canvas=$('mesh');canvas.hidden=!$('landmarks').checked;
    if(!canvas.hidden){canvas.width=320;canvas.height=240;const c=canvas.getContext('2d');c.clearRect(0,0,320,240);for(const t of multimodal.tracks){c.fillStyle=t.speaking?'#16804a':'#b88a50';for(const p of t.points){c.beginPath();c.arc((1-p.x)*320,p.y*240,1,0,Math.PI*2);c.fill();}c.fillText(t.id,(1-t.box.x-t.box.w)*320,t.box.y*240);}}
   };
  }
  cameraStream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,frameRate:{ideal:CAMERA_FPS,max:CAMERA_FPS},facingMode:'user'},audio:false});
  cameraGeneration++;lastVideo=-1;multimodal.setCamera('on');
  videoFrameAt=null;const frameTicket=cameraGeneration;
  const stamp=(now,metadata)=>{if(frameTicket!==cameraGeneration)return;videoFrameAt=Number.isFinite(metadata.captureTime)?metadata.captureTime:now;$('video').requestVideoFrameCallback?.(stamp);};
  $('video').requestVideoFrameCallback?.(stamp);
  cameraStream.getVideoTracks()[0].onended=()=>{cameraStream=null;cameraGeneration++;multimodal.setCamera('unavailable');gaze.resetTracking();$('camera').textContent='Camera off';};
  $('video').srcObject=cameraStream;await $('video').play();$('video').hidden=false;$('camera').textContent='Camera on';$('camera-status').textContent='Tracking locally in CPU worker. Images and face descriptors stay in this browser.';
  return true;
 }catch(e){cameraStream?.getTracks().forEach(t=>t.stop());cameraStream=null;landmarker?.terminate();landmarker=null;cameraBusy=false;multimodal.setCamera('unavailable');$('camera').textContent='Enable camera';$('camera-status').textContent='Camera tracking unavailable: '+e.message;$('follow-status').textContent='Camera permission unavailable. Click Enable camera to retry; audio still works.';return false;}
}
async function observeFace(){
 const video=$('video');if(cameraBusy||!cameraStream||!landmarker||video.readyState<2||video.currentTime===lastVideo)return;
 cameraBusy=true;lastVideo=video.currentTime;const generation=cameraGeneration,now=performance.now();
 // Some browsers stop presentation callbacks for a collapsed preview while
 // currentTime keeps advancing. Do not reuse that frozen timestamp forever.
 const valid=Number.isFinite(videoFrameAt)&&videoFrameAt<=now+50&&now-videoFrameAt<750&&videoFrameAt>(window.robotFace.cameraStats.lastFrameAt??-Infinity);
 const at=valid?videoFrameAt:now;window.robotFace.cameraStats.lastFrameAt=at;
 window.robotFace.cameraStats.timestampFallback=!valid;
 try{const frame=await createImageBitmap(video);if(generation!==cameraGeneration){frame.close();cameraBusy=false;return;}landmarker.postMessage({type:'frame',frame,at,generation},[frame]);}catch{cameraBusy=false;}
}
let lastFace=0,lastAnimation=0;
function animate(t){
 if(eyeContact?.state.active)avatarActions.clear();
 if(eyeContact?.state.frozen){const f=eyeContact.state.frozen;rig.update(f.input,lastAnimation?t-lastAnimation:16.7,f.time);lastAnimation=t;requestAnimationFrame(animate);return;}
 if(t-lastFace>=FACE_INTERVAL_MS){try{observeFace();}catch{}lastFace=t;}
 const cue=timeline.at(client.ctx?.state==='running'?client.clock?.position():null);
 const preview=$('expression').value,acting=preview==='auto'&&cue;
 const gazeTarget=cameraStream&&$('mirror').checked?gaze.read(t):null;
 const follows=!!gazeTarget,controls=follows?{x:deskView?0:gazeTarget.pan/52,y:deskView?0:-gazeTarget.tilt/15,roll:0}:{};
 const idle=deskView?rig.idlePose():pose();
 const target=preview!=='auto'?pose(preview):acting?cue.pose:follows?(deskView?mix(idle,pose('amusement',.65),Math.min(.65,face.smile||0)):pose('amusement',Math.min(.5,face.smile||0))):idle;
 const chosen=mix(pose(),target,Number($('face-intensity').value));
 const description=preview!=='auto'?'Silent preview: '+preview.replaceAll('_',' '):acting?
  `${cue.emotion.replaceAll('_',' ')}${cue.secondary?' + '+cue.secondary.replaceAll('_',' '):''} · sentence ${cue.index+1} · Luna SCRIPT · estimated sentence timing`:
  'Luna’s sentence-level acting directions drive the face. Audio only moves the mouth.';
 if($('acting-status').textContent!==description)$('acting-status').textContent=description;
 const automaticBlink=Math.pow(Math.max(0,Math.cos(t/900)),160)*.85;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 let action=avatarActions.frame(t,chosen,{position:client.ctx?.state==='running'?client.clock?.position():null,busy:speaking||replyPending,reduced,hidden:document.hidden,calibrating:!!eyeContact?.state.active});
 const pet=petting?.frame(t,action.pose,reduced);if(pet)action=pet;
 rig.update({...controls,...action.controls,gazeTarget:deskView?gazeTarget:null,pose:action.pose,actionMotion:action.motion,actionEyes:action.eyes,
  blinkLeft:Math.max(action.controls.blinkLeft||0,controls.blinkLeft||0,automaticBlink),blinkRight:Math.max(action.controls.blinkRight||0,controls.blinkRight||0,automaticBlink),audioLevel:Math.min(1,client.level()*7)},lastAnimation?t-lastAnimation:16.7,reduced?0:t);lastAnimation=t;
 requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
organisePlayground();
$('mirror').onchange=()=>{try{localStorage.setItem('robby.followPosition',String($('mirror').checked));}catch{}if($('mirror').checked){rig.center?.();void startCamera();}};
$('camera-auto').onchange=()=>{try{localStorage.setItem('robby.cameraAuto',String($('camera-auto').checked));}catch{}if($('camera-auto').checked)void startCamera();};
$('follow-center').onclick=()=>{if(eyeContact?.state.active)eyeContact.finish();rig.center?.();gaze.resetTracking();$('mirror').checked=true;if($('desk-manual'))$('desk-manual').checked=false;if($('desk-tour'))$('desk-tour').checked=false;$('mirror').onchange();$('follow-status').textContent='Eye contact resumed. Waiting for fresh camera landmarks…';};
$('quick-calibrate').onclick=()=>{const panel=document.querySelector('.gaze-controls');if(panel){panel.open=true;$('gaze-start').click();}};
window.robotFace.cameraControls={ensure:startCamera,stop:stopCamera,get active(){return !!cameraStream;}};
if($('camera-auto').checked)void startCamera();else $('camera').textContent='Enable camera';
