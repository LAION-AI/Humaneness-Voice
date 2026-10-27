// Last conversational context and measured stage timings, without guessing zeros.
const ids=['asr','speaker','reaction','bridge','llm','tts','reply','buffer','sound','events'];
const finite=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
const ms=n=>finite(n)?(n>0&&n<1?'<1 ms':`${Math.round(n)} ms`):'—';
export class LiveOverview{
 constructor(){
  this.values={};this.turn=null;this.pending=false;this.hasInput=false;
  const quick=document.getElementById('quick-controls'),options=document.createElement('details');options.id='conversation-options';options.className='compact-conversation-options';
  const optionsTitle=document.createElement('summary');optionsTitle.textContent='Speech pause & interruption options';options.append(optionsTitle,document.getElementById('speech-pause-controls'));
  const interaction=document.querySelector('.interaction-controls'),primary=document.getElementById('allow-proactive').closest('label');
  const primaryRow=document.createElement('div');primaryRow.className='quick-interrupt-switch';primaryRow.append(primary);quick.append(primaryRow);
  options.append(document.getElementById('voice-caption-controls'));options.append(interaction);quick.append(options);
  const aside=document.querySelector('aside');
  const recent=document.createElement('section');recent.id='live-conversation';recent.setAttribute('aria-label','Latest conversation beside Robby');
  recent.innerHTML='<h2>With Robby, right now</h2><article class="recent-card asr"><h3>ASR · live preview</h3><p id="recent-asr" class="recent-copy">Waiting for speech…</p></article><article class="recent-card user"><h3 id="recent-user-label">You · last turn</h3><p id="recent-user" class="recent-copy">Say something or type a message.</p></article><article class="recent-card robot"><h3>Robby · latest reply</h3><p id="recent-robby" class="recent-copy">I’m ready to chat with you.</p></article>';
  const playbackStatus=document.createElement('p');playbackStatus.id='recent-playback-status';playbackStatus.setAttribute('role','status');
  playbackStatus.textContent='No spoken reply yet.';recent.querySelector('.robot').append(playbackStatus);
  const history=document.createElement('details');history.className='settings chat-history';history.id='chat-history';
  const title=document.createElement('summary');title.textContent='Full conversation history';history.append(title,document.querySelector('.chat-heading'),document.getElementById('chat'));
  history.ontoggle=()=>{if(history.open){const chat=document.getElementById('chat');chat.scrollTop=chat.scrollHeight;}};
  const form=document.getElementById('form');aside.prepend(history);
  const row=document.createElement('div');row.className='conversation-row';
  const robotColumn=document.createElement('div');robotColumn.className='robot-column';
  const contextColumn=document.createElement('div');contextColumn.className='conversation-context';
  const avatar=document.getElementById('avatar'),under=document.querySelector('.under-avatar');avatar.before(row);
  robotColumn.append(avatar,under);contextColumn.append(recent,form);row.append(robotColumn,contextColumn);
  const people=document.querySelector('.people-panel');people.open=false;aside.append(people);
  const panel=document.createElement('section');panel.id='latency-dashboard';panel.setAttribute('aria-label','Latency of the current reply');
  const labels={asr:'ASR · final check',speaker:'Speaker ID · wait',reaction:'Local quick reply',bridge:'Informed bridge',llm:'LLM · first sentence',tts:'TTS · first PCM',reply:'Reply → first PCM',buffer:'Playback lead',sound:'Sound caption · GPU request',events:'Sound events · CPU request'};
  panel.innerHTML='<div class="latency-heading"><h2>Response timing</h2><span id="latency-status" role="status">Waiting for a turn</span></div><div class="latency-grid">'+ids.map(id=>`<div class="latency-cell"><span>${labels[id]}</span><strong id="latency-${id}">—</strong></div>`).join('')+'</div><details><summary>Timing details</summary><p id="latency-details">These are measured server / browser timings. Stages overlap, so the values must not be added. A dash means not measured yet.</p></details>';
  document.querySelector('.under-avatar').after(panel);
 }
 copy(id,text){const p=document.getElementById(id);p.textContent=text;p.scrollTop=p.scrollHeight;}
 clear(){this.values={};this.turn=null;this.pending=false;this.hasInput=false;this.kind=null;this.copy('recent-asr','Waiting for speech…');this.copy('recent-user','Say something or type a message.');this.copy('recent-robby','I’m ready to chat with you.');document.getElementById('recent-user-label').textContent='You · last turn';this.paint();}
 begin(kind){this.values={};this.turn=null;this.pending=true;this.hasInput=true;this.kind=kind;this.audioReceived=false;this.paint();}
 typed(text){this.copy('recent-user',text);this.copy('recent-asr','Typed input · no ASR needed.');this.copy('recent-robby','Preparing a reply…');document.getElementById('recent-user-label').textContent='You · typed';}
 audio(m){if(m.first){this.audioReceived=true;this.copy('recent-playback-status','Audio received · playback scheduled.');this.values.buffer=m.playback_lead_ms;this.paint();}}
 playbackEnded(){if(this.audioReceived)this.copy('recent-playback-status','Current audio buffer played.');}
 message(m){
  if(m.type==='sound_caption'){this.lastSound=m;this.values.sound=m.status==='available'?m.request_ms:null;this.values.events=m.sound_events?.status==='available'?m.sound_events.request_ms:null;this.paint();}

  if(['asr_preview','partial'].includes(m.type)&&m.text?.trim())this.copy('recent-asr',m.text);
  if(m.type==='latency_start'){
   this.begin(m.input?.kind||'unknown');this.turn=m.reply_id;const p=m.input||{};
   this.values.asr=p.asr?.transcribe_ms;this.values.pause=p.asr?.observed_quiet_ms;this.values.asr_model=p.asr?.model;
   this.values.events=this.lastSound?.sound_events?.status==='available'?this.lastSound.sound_events.request_ms:null;this.values.sound=p.sound_caption_ms??(this.lastSound?.status==='available'?this.lastSound.request_ms:null);this.values.caption=p.voice_caption_ms;this.values.caption_status=p.voice_caption_status;this.values.speaker=p.speaker_assignment_ms;this.values.speaker_background=p.speaker_prefetch_compute_ms;
   this.values.speaker_cached=p.speaker_prefetch_used;this.values.queue=p.sentence_queue_ms;this.paint();
  }
  if(m.type==='asr_finalizing')document.getElementById('latency-status').textContent='Recognizing your next words…';
  if(m.type==='asr_sentence'){
   this.copy('recent-asr',m.text);this.copy('recent-user',m.text);
   document.getElementById('recent-user-label').textContent=m.queued?'You · queued next turn':'You · recognized';
  }
  if(m.type==='endpoint'){
   this.copy('recent-user',m.transcript||'');
   document.getElementById('recent-user-label').textContent=(m.speaker_name||'You')+' · last turn';
  }
  if(m.type==='reply'){
   if(!this.pending)this.begin(m.avatar_event?'avatar':'reply');
   this.turn=m.reply_id;this.copy('recent-robby',m.text||'');this.consume(m.timing||{});
   if(!this.audioReceived)this.copy('recent-playback-status','Text ready · waiting for speech audio.');
  }
  if(m.type==='latency_update'||m.type==='pipeline_metrics'){
   if(this.turn&&m.reply_id&&this.turn!==m.reply_id)return;
   this.consume(m.metrics||{});
  }
  if(m.type==='turn_done'){this.pending=false;document.getElementById('latency-status').textContent='Latest response · generation complete';}
  if(['barge','reply_cancelled','stop_command'].includes(m.type)){this.copy('recent-playback-status',this.audioReceived?'Speech interrupted · some text may not have been heard.':'Reply cancelled before speech playback.');this.audioReceived=false;this.pending=false;document.getElementById('latency-status').textContent='Interrupted · last measured timings retained';}
  if(m.type==='error'){this.copy('recent-playback-status','Speech failed · this text may not have been played.');this.audioReceived=false;this.pending=false;document.getElementById('latency-status').textContent='Reply error · available timings retained';}
 }
 consume(metrics){
  if(metrics.quick_reply_status){this.values.reaction_status=metrics.quick_reply_status;document.getElementById('quick-reply-status').textContent=metrics.quick_reply_status.replaceAll('_',' ')+(metrics.quick_reply_words?` · ${metrics.quick_reply_words} words`:'');}
  if(metrics.informed_bridge_status)this.values.bridge_status=metrics.informed_bridge_status;
  if(finite(metrics.answer_audio_ready_ms))this.values.answer_ready=metrics.answer_audio_ready_ms;
  const fields={reaction:'quick_reply_generation_ms',bridge:'informed_bridge_generation_ms',answer:'answer_first_audio_ms',llm:'llm_first_sentence_ms',ttft:'llm_ttft_ms',tts:'tts_first_audio_ms',reply:'first_audio_ms',llm_wall:'llm_total_ms',rtf:'generation_rtf'};
  for(const [key,name] of Object.entries(fields))if(finite(metrics[name]))this.values[key]=metrics[name];
  if(finite(metrics.tool_director_ms)&&!finite(this.values.llm))this.values.llm=metrics.tool_director_ms;
  if(metrics.llm_model)this.values.model=metrics.llm_model;
  if(metrics.llm_actual_provider||metrics.llm_provider)this.values.provider=metrics.llm_actual_provider||metrics.llm_provider;
  const stages=metrics.tts_server_stages||[];
  if(stages.some(s=>finite(s.generation_ms)))this.values.tts_gpu=stages.reduce((sum,s)=>sum+(finite(s.generation_ms)?s.generation_ms:0),0);
  this.paint();
 }
 paint(){
  for(const id of ids)document.getElementById('latency-'+id).textContent=id==='asr'&&this.kind==='typed'?'Typed':ms(this.values[id]);
  const v=this.values;document.getElementById('latency-status').textContent=this.pending?'Current response · updating live':'Latest measured response';
  document.getElementById('latency-details').textContent=[
   `Model: ${v.model||'waiting'}${v.provider?' · '+v.provider:''}`,
   `LLM first token: ${ms(v.ttft)}; LLM stream wall time: ${ms(v.llm_wall)}`,
   `TTS job wall time (overlapping; completed sentences): ${ms(v.tts_gpu)}; generation RTF: ${finite(v.rtf)?v.rtf.toFixed(2):'—'}`,
   `Local reaction: ${v.reaction_status||'not measured'}; first useful answer PCM: ${ms(v.answer)}`,
   `Informed bridge: ${v.bridge_status||'not measured'}; main PCM ready in buffer: ${ms(v.answer_ready)}`,
   `Environmental sound: capture ${ms(this.lastSound?.capture_ms)}; GPU inference ${ms(this.lastSound?.inference_ms)}; worker request ${ms(this.lastSound?.request_ms)}. Runs alongside ASR; can be skipped.`,
   `Optional tone caption: ${ms(v.caption)} · ${v.caption_status||'not measured'}`,
   `Speech pause observed: ${ms(v.pause)}; input turn queue: ${ms(v.queue)}`,
   `Speaker background embedding: ${ms(v.speaker_background)}${v.speaker_cached?' (prefetched; main card shows only the remaining wait)':''}`,
   'ASR is the pause-final pass; TTS first PCM includes server transport/decoding. Reply → first PCM excludes ASR/speaker work. Playback lead is browser scheduling, not measured loudspeaker latency. Stages overlap: do not add these values.'
  ].join('\n');
 }
}
