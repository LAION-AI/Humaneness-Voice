// One existing microphone and playback context; no hidden capture or face-ID.
const read=(key,fallback)=>{try{const v=localStorage.getItem(key);return v===null?fallback:JSON.parse(v);}catch{return fallback;}};
const save=(key,value)=>{try{localStorage.setItem(key,JSON.stringify(value));}catch{}};
export class EnvironmentControls{
 constructor(client){
  this.client=client;this.attempted=false;this.ready=false;this.greeted=false;
  this.auto=read('robby.autoStart',true)===true;
  this.enabled=read('robby.environment',true)===true;
  this.voices=read('robby.environmentVoices',true)===true;
  this.events=read('robby.soundEvents',true)===true;
  this.impressions=read('robby.voiceImpressions',false)===true;
  if(read('robby.soundThresholdVersion',0)!==2){if(read('robby.soundProminence',12)===12)save('robby.soundProminence',8);save('robby.soundThresholdVersion',2);}
  const raw=read('robby.soundProminence',8);this.prominence=Number.isFinite(raw)?Math.min(24,Math.max(6,raw)):8;
  const banner=document.createElement('section');banner.id='auto-start-banner';banner.className='environment-banner';
  banner.innerHTML='<label><input id="robby-auto-start" type="checkbox"> Start Robby automatically</label><span id="auto-start-status" role="status"></span><button id="enable-robby" type="button" hidden>Enable microphone &amp; sound</button><button id="disable-robby" type="button">Turn Robby off</button>';
  document.getElementById('avatar').after(banner);
  const panel=document.createElement('details');panel.className='settings environment-settings';panel.id='environment-settings';
  panel.innerHTML='<summary>Proactive listening &amp; sounds</summary><label><input id="environment-enabled" type="checkbox"> Notice prominent sounds</label><label><input id="environment-voices" type="checkbox"> Ask about unfamiliar foreground voices</label><label><input id="sound-events-enabled" type="checkbox"> AudioSet event candidates · CPU, in parallel</label><label><input id="preserve-sounds" type="checkbox"> Preserve music and environmental sounds at the microphone</label><label><input id="environment-impressions" type="checkbox"> Include uncertain vocal age / gender impressions</label><label for="sound-prominence">Sound must exceed background by <output id="sound-prominence-value"></output> dB</label><input id="sound-prominence" type="range" min="6" max="24" step="1"><p>Default: 8 dB above background. Higher values are less sensitive. Preserve sounds disables browser noise suppression and automatic gain; echo cancellation stays on. Quiet sounds and Robby’s own playback are filtered out. Prominent music may receive one curious comment, with a three-minute music cooldown. Replies wait for a pause. Ask “Was ist das hier?” or “What is that sound?” and play the sound immediately: Robby listens for about 2.4 seconds before answering. A periodic check also notices music already playing when the microphone starts. Voice impressions describe how a voice sounds; they do not establish identity, age or gender.</p><p id="environment-status" role="status">Waiting for the microphone.</p><p id="environment-caption"></p><p>The existing microphone is processed on this server. No environmental recordings are saved. Short, uncertain captions can enter the selected language model’s context. This is not a safety alarm. Camera settings are unchanged.</p><p><a href="/static/environment-report.html" target="_blank" rel="noopener">Measurements &amp; reproducible tests</a></p>';
  document.querySelector('aside').append(panel);
  const $=id=>document.getElementById(id);this.status=$('auto-start-status');this.enable=$('enable-robby');
  $('sound-events-enabled').checked=this.events;
  $('robby-auto-start').checked=this.auto;$('environment-enabled').checked=this.enabled;$('environment-voices').checked=this.voices;$('environment-impressions').checked=this.impressions;
  $('preserve-sounds').checked=read('robby.preserveSounds',true)!==false;
  $('preserve-sounds').onchange=async()=>{const preserve=$('preserve-sounds').checked;save('robby.preserveSounds',preserve);try{await this.client.stream?.getAudioTracks()[0]?.applyConstraints({echoCancellation:true,noiseSuppression:!preserve,autoGainControl:!preserve});}catch{this.status.textContent='Microphone settings will apply on the next connection.';}};
  $('sound-prominence').value=this.prominence;$('sound-prominence-value').textContent=this.prominence;
  $('robby-auto-start').onchange=()=>{this.auto=$('robby-auto-start').checked;save('robby.autoStart',this.auto);if(this.auto){this.attempted=false;void this.start();}else this.off();};
  this.enable.onclick=()=>{this.attempted=false;void this.start();};
  $('disable-robby').onclick=()=>this.off();
  for(const [id,field,key] of [['sound-events-enabled','events','robby.soundEvents'],['environment-enabled','enabled','robby.environment'],['environment-voices','voices','robby.environmentVoices'],['environment-impressions','impressions','robby.voiceImpressions']]){
   $(id).onchange=()=>{this[field]=$(id).checked;save(key,this[field]);this.configure();};
  }
  $('sound-prominence').oninput=()=>{$('sound-prominence-value').textContent=$('sound-prominence').value;};
  $('sound-prominence').onchange=()=>{this.prominence=Number($('sound-prominence').value);save('robby.soundProminence',this.prominence);this.configure();};
  this.status.textContent=this.auto?'Starts when the server and browser permissions are ready.':'Automatic start is off.';
 }
 configure(){this.client.send('environment:'+JSON.stringify({enabled:this.enabled,voices:this.voices,impressions:this.impressions,events:this.events,prominence_db:this.prominence}));}
 poll(ready){this.ready=ready;if(ready&&this.auto&&!this.attempted)void this.start();}
 async start(){
  if(!this.ready)return;this.attempted=true;this.enable.hidden=true;this.status.textContent='Starting microphone & sound…';
  try{await this.client.start(true);this.status.textContent='Microphone active. Turn Robby off to stop listening and playback.';}
  catch(e){this.status.textContent='Browser permission needed: '+e.message;this.enable.hidden=false;}
 }
 off(){
  this.auto=false;this.attempted=true;save('robby.autoStart',false);
  document.getElementById('robby-auto-start').checked=false;
  this.client.send('environment:'+JSON.stringify({enabled:false}));this.client.stop();
  this.status.textContent='Robby is off. Microphone and playback stopped. Automatic start is disabled.';this.enable.hidden=false;
 }
 message(m){
  if(m.type==='ready'){
   this.configure();
   if(this.auto&&!this.greeted&&this.client.ctx?.state==='running'){
    this.greeted=true;this.client.send('environment:'+JSON.stringify({welcome:true}));
   }
  }
  if(m.type==='environment_settings')document.getElementById('environment-status').textContent=m.enabled?'Listening alongside ASR · prominent sounds only · no saved recordings.':'Proactive sound reactions are off.';
  if(m.type==='sound_caption'){
   document.getElementById('environment-caption').textContent=m.caption?'Unconfirmed sound: '+m.caption+(m.sound_events?.status==='available'?' · Event candidates: '+m.sound_events.candidates.slice(0,3).map(c=>c.label+' '+c.score.toFixed(2)).join(' / '):''):'';
   document.getElementById('environment-status').textContent=`Sound caption: ${m.status} · ${Math.round(m.request_ms||0)} ms${m.kind?' · '+m.kind:''}${m.eligible?' · waiting for a conversational pause':' · no proactive reply'}`;
  }
  if(m.type==='environment_observation')document.getElementById('environment-status').textContent=(m.kind==='welcome'?'Welcome':m.kind==='music'?'Possible music':m.kind==='unfamiliar_voice'?'Possible unfamiliar voice':'Prominent sound')+' · '+m.status.replaceAll('_',' ');
 }
}
