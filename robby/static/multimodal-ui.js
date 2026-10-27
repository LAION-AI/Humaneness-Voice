import {VisualSpeakers,People} from './visual-speakers.js';
import {PeopleView,faceCanvas} from './people-view.js';
const DEFAULTS={ignoreOffscreenSpeakers:false,lookAtActiveSpeaker:true,greetUnknownSpeakers:true,faceIdEnabled:false};
const KEYS=Object.keys(DEFAULTS),SETTINGS='robby.multimodal.settings.v2';
export class MultimodalUI{
 constructor(client,host){
  this.client=client;this.visual=new VisualSpeakers();let storage=null;try{storage=localStorage;}catch{}
  this.people=new People(storage);this.storage=storage;this.settings={...DEFAULTS};
  // Migrate gaze/greeting preferences, but do not inherit the old forced-on
  // camera gate. Existing users receive the new audio-first defaults once.
  try{const old=JSON.parse(storage?.getItem('robby.multimodal.settings.v1')||'{}');for(const k of ['lookAtActiveSpeaker','greetUnknownSpeakers'])if(typeof old[k]==='boolean')this.settings[k]=old[k];}catch{}
  try{const old=JSON.parse(storage?.getItem(SETTINGS)||'{}');for(const k of KEYS)if(typeof old[k]==='boolean')this.settings[k]=old[k];}catch{}
  this.pending=new Map();this.metrics={landmarkMs:[],embeddingMs:[],embeddingCalls:0};this.camera='off';this.generation=0;this.tracks=[];
  const panel=this.panel=document.createElement('section');panel.className='speaker-tuning multimodal-controls';panel.setAttribute('aria-label','Visible speakers');
  const heading=document.createElement('strong');heading.textContent='Visible speakers · local camera recognition';panel.append(heading);
  const quick=document.createElement('section');quick.className='camera-recognition-options';quick.setAttribute('aria-label','Optional camera recognition');
  const note=document.createElement('p');note.textContent='Voice recognition stays on. The camera follows your position without checking who you are.';quick.append(note);
  (document.getElementById('quick-controls')||panel).append(quick);
  for(const [key,label] of [[KEYS[0],'Require visible mouth movement to answer (face VAD)'],[KEYS[1],'Look at the active speaker'],[KEYS[2],'Greet unknown speakers'],['faceIdEnabled','Recognize people by webcam Face ID (optional)']]){
   const row=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.id=key;input.checked=this.settings[key];
   input.onchange=()=>{this.settings[key]=input.checked;if(key==='faceIdEnabled'&&!input.checked)this.stopFaceIdentity();try{storage?.setItem(SETTINGS,JSON.stringify(this.settings));}catch{}this.sendSettings();this.render();};row.append(input,' '+label);(key==='ignoreOffscreenSpeakers'||key==='faceIdEnabled'?quick:panel).append(row);
  }
  this.status=document.createElement('p');this.status.id='visual-speaker-status';this.status.setAttribute('role','status');panel.append(this.status);
  this.list=document.createElement('p');this.list.id='visual-person-list';panel.append(this.list);
  const help=document.createElement('p');help.textContent='Face VAD and webcam Face ID are OFF by default, independently of camera tracking and acoustic speaker recognition. Enable face VAD only to require visible mouth movement before answering. Face ID is a separate opt-in. Face/voice cues are not authentication.';panel.append(help);
  const label=document.createElement('label'),consent=this.consent=document.createElement('input');consent.type='checkbox';consent.id='remember-biometric-profiles';consent.checked=this.people.consent;
  label.append(consent,' Remember face + voice profiles on this device (requires everyone’s permission)');panel.append(label);
  consent.onchange=()=>{
   if(consent.checked&&!confirm('Continue only with explicit permission from EVERY person whose face, name and voice profile will be saved. Profiles are stored unencrypted in this browser on this device. They are never sent to Luna. Continue?'))consent.checked=false;
   this.people.setConsent(consent.checked);consent.checked=this.people.consent;
   if(this.people.storageError)this.faceError='Browser storage unavailable. Session memory only; clear this site’s browser data if saved profiles could not be removed.';
   if(consent.checked)this.send({type:'export',consent:true});
  };
  const forget=document.createElement('button');forget.type='button';forget.className='text-button';forget.id='forget-multimodal-people';forget.textContent='Forget all face + voice profiles';forget.onclick=()=>{
   this.people.clear();consent.checked=false;this.pending.clear();this.visual.clear();this.generation++;if(this.people.storageError)this.faceError='Storage access denied: clear this site’s browser data to remove saved profiles.';this.send({type:'forget'});this.render();window.dispatchEvent(new Event('robby-forget-people'));};panel.append(forget);
  host.append(panel);const overview=document.getElementById('people-overview');if(overview)this.peopleView=new PeopleView(this,overview);this.render();
  // Keep recovery beside the rejected transcript, not buried in Advanced.
  const recovery=this.recovery=document.createElement('button');recovery.type='button';recovery.className='text-button';recovery.id='visual-filter-recovery';recovery.hidden=true;
  recovery.textContent='Use voice recognition without camera filtering';
  recovery.onclick=()=>{this.settings.ignoreOffscreenSpeakers=false;const input=document.getElementById('ignoreOffscreenSpeakers');input.checked=false;input.onchange();
   recovery.hidden=true;document.getElementById('asr-status').textContent='Camera filter OFF; your voice recognition setting is unchanged. Please say your introduction again. Camera and eye contact stay on.';this.render();};
  document.getElementById('asr-status')?.after(recovery);
 }
 send(data){this.client.send('visual:'+JSON.stringify(data));}
 sendSettings(){this.send({type:'settings',...this.settings});}
 stopFaceIdentity(){
  this.generation++;this.worker?.terminate();this.worker=null;this.embeddingBusy=false;this.pending.clear();this.faceError=null;
  for(const t of this.visual.tracks.values()){t.embedding=null;t.personId=null;t.referenceImage=null;t.pendingReferenceImage=null;t.faceDistance=null;}
 }
 ready(){this.sendSettings();this.sendFrame();const profiles=[...this.people.profiles.values()].filter(p=>p.voice).map(p=>({...p.voice,person_id:p.id,name:p.name}));if(profiles.length)this.send({type:'import',session:true,profiles});}
 setCamera(state){this.camera=state;this.generation++;this.visual.clear();this.tracks=[];this.pending.clear();this.sendFrame();this.render();}
 disconnected(){this.pending.clear();this.generation++;}
 frame(result,at,video,ms){
  this.metrics.landmarkMs.push(ms);this.metrics.landmarkMs=this.metrics.landmarkMs.slice(-300);
  this.tracks=this.visual.update(result.faceLandmarks||[],at,this.settings.ignoreOffscreenSpeakers||this.settings.faceIdEnabled);this.sendFrame(at);
  this.peopleView?.updateVideo(video,this.tracks,performance.now());
  const now=performance.now(),candidate=this.tracks.find(t=>(!t.embedding||now-(t.embeddingAt||0)>20000)&&t.box.w*video.videoWidth>=75&&t.box.h*video.videoHeight>=90&&now-t.born>500&&now>(t.retryAt||0)&&now>(t.ambiguousUntil||0));
  if(this.settings.faceIdEnabled&&candidate&&!this.embeddingBusy&&now-(this.lastEmbedding??-Infinity)>1500)this.embed(candidate,video);
  const audioTrack=this.visual.tracks.get(this.audioTrack);
  let primary=this.visual.tracks.get(this.primaryTrack);
  if(!primary||at-primary.seen>500){primary=[...this.tracks].sort((a,b)=>b.box.w*b.box.h-a.box.w*a.box.h)[0];this.primaryTrack=primary?.id;}
  const followed=this.settings.lookAtActiveSpeaker&&audioTrack&&at-(this.audioActiveAt||0)<1300&&at-audioTrack.seen<500?audioTrack:primary;
  this.followedTrack=followed?.id;
  this.render();if(!followed)return result.faceLandmarks?.length===1?result:null;
  const i=(result.faceLandmarks||[]).indexOf(followed.points);
  return {...result,faceLandmarks:[followed.points],faceBlendshapes:[result.faceBlendshapes?.[i]],facialTransformationMatrixes:[result.facialTransformationMatrixes?.[i]]};
 }
 sendFrame(at=performance.now()){
  const audio=this.client.audioTimeAt(at);if(this.camera==='on'&&audio===null)return;
  this.send({type:'frame',camera:this.camera,audio_s:audio,faces:this.visual.snapshot(at)});
 }
 render(){
  this.status.textContent=!this.settings.ignoreOffscreenSpeakers?'Face VAD OFF · voice recognition decides who may speak. Camera '+this.camera+' · position tracking is independent.':this.camera!=='on'?'Camera '+this.camera+' · audio-only fallback. Turn on the camera for visible-speaker filtering.':
   this.tracks.length?this.tracks.map(t=>`${this.people.profiles.get(t.personId)?.name||t.id}: ${t.speaking?'mouth moving ('+Math.round(t.confidence*100)+'%)':'not speaking'}`).join(' · '):'Camera on · no visible face. Offscreen speech is ignored unless disabled above.';
  this.list.textContent='Webcam Face ID '+(this.settings.faceIdEnabled?'ON':'OFF')+' · Stored optional profiles: '+([...this.people.profiles.values()].map(p=>p.name).join(', ')||'none')+' · '+(this.people.consent?'saved on this device with consent':'session memory only')+(this.faceError?' · Face-ID: '+this.faceError:'');
  this.peopleView?.render();
 }
 async embed(t,video){
  if(!this.settings.faceIdEnabled)return;
  this.embeddingBusy=true;this.lastEmbedding=performance.now();t.retryAt=this.lastEmbedding+10000;
  try{
   if(!this.worker){this.worker=new Worker('/static/face-id-worker.js');this.worker.onmessage=e=>this.embedded(e.data);this.worker.onerror=()=>{this.embeddingBusy=false;this.faceError='Face-ID worker unavailable';};}
   const canvas=faceCanvas(video,t.points),ctx=canvas.getContext('2d');
   t.pendingReferenceImage=canvas.toDataURL('image/jpeg',.65);
   const image=ctx.getImageData(0,0,150,150).data,pixels=new Uint8Array(150*150*3);for(let i=0,j=0;i<image.length;i+=4){pixels[j++]=image[i];pixels[j++]=image[i+1];pixels[j++]=image[i+2];}
   this.worker.postMessage({id:t.id,generation:this.generation,identityEpoch:t.identityEpoch||0,pixels:pixels.buffer},[pixels.buffer]);
  }catch(e){this.embeddingBusy=false;this.faceError=e.message;}
 }
 embedded(d){this.embeddingBusy=false;if(!this.settings.faceIdEnabled||d.generation!==this.generation)return;const t=this.visual.tracks.get(d.id);if(!t||performance.now()<(t.ambiguousUntil||0)||d.identityEpoch!==(t.identityEpoch||0))return;
  if(d.error){this.faceError=d.error;return;}
  this.metrics.embeddingCalls++;this.metrics.embeddingMs.push(d.ms);this.metrics.embeddingMs=this.metrics.embeddingMs.slice(-100);t.embedding=d.face;t.embeddingAt=performance.now();t.referenceImage=t.pendingReferenceImage;
  const person=this.people.match(d.face);if(person){t.personId=person.id;t.faceDistance=Math.hypot(...d.face.map((v,i)=>v-person.face[i]));this.send({type:'identify',track_id:t.id,person_id:person.id,embedding_age_ms:0});}
  const binding=this.pending.get(t.id);if(binding)this.bind(binding);this.render();
 }
 bind(m){if(!this.settings.faceIdEnabled)return;const t=this.visual.tracks.get(m.track_id);if(!t)return;
  if(!t.embedding){this.pending.set(t.id,m);return;}
  const recognized=this.people.match(t.embedding);
  if(recognized&&recognized.id!==m.person_id){this.faceError='Face and voice identities conflict; introduce yourself again.';return;}
  if(!this.people.profiles.has(m.person_id)&&this.people.consent){this.people.setConsent(false);this.consent.checked=false;this.faceError='New person: saving paused. Obtain everyone’s permission and enable Remember again to save all current profiles.';}
  if(this.people.bind(m.person_id,m.name,t.embedding,null,t.referenceImage)){t.personId=m.person_id;this.pending.delete(t.id);this.send({type:'identify',track_id:t.id,person_id:m.person_id,embedding_age_ms:performance.now()-(t.embeddingAt||0)});this.send({type:'export',session:true});}this.render();
 }
 message(m){
  if(m.type==='ignored'&&m.source==='visual_filter'){
   this.recovery.hidden=false;const details=this.recovery.closest('details');if(details)details.open=true;
  }
  if(m.type==='endpoint'||m.type==='ready')this.recovery.hidden=true;
  this.peopleView?.message(m);
  if(m.type==='visual_check')for(const t of this.tracks)if(t.personId&&performance.now()-(t.embeddingAt||0)>3500){t.embedding=null;t.retryAt=0;}
  if(m.type==='visual_active'){
   const now=performance.now();
   if(this.audioTrack===m.track_id)this.audioActiveAt=now;
   else if(this.audioCandidate?.id!==m.track_id)this.audioCandidate={id:m.track_id,since:now};
   else if(now-this.audioCandidate.since>450&&now-(this.lastLookSwitch||0)>700){this.audioTrack=m.track_id;this.audioActiveAt=now;this.lastLookSwitch=now;}
  }
  if(m.type==='visual_person')this.bind(m);
  if(m.type==='visual_greeting'){this.audioTrack=m.track_id;this.audioActiveAt=performance.now();}
  if(m.type==='visual_profiles'){for(const p of m.profiles||[]){const known=this.people.profiles.get(p.person_id);if(known)known.voice=p;}try{this.people.save();}catch{}this.render();}
  if(m.type==='visual_warning')this.status.textContent=m.text;
 }
}
