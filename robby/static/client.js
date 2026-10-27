// TYGER wire and capture worklet adapted from nineninesix-ai/tyger-s2s (Apache-2.0).
// One AudioContext owns capture AND playback; tags represent DAC playback, not arrival.
import {PlaybackClock} from './clock.js';
export function playbackStart(context,next,firstReply,reserve=.45){
 const deviceLead=Math.max(128/context.sampleRate,(context.baseLatency||0)*1.25);
 // A short first codec block can run out before the next TTS chunk arrives.
 // This is a real scheduling reserve, NOT a fabricated Second Ear offset.
 return Math.max(next,context.currentTime+deviceLead+(firstReply ? reserve : 0));
}
const WORKLET=`
class Mic16k extends AudioWorkletProcessor {
 constructor(){super();this.step=sampleRate/16000;this.pos=-1;this.prev=0;this.out=new Float32Array(320);this.n=0;this.t0=0;}
 process(inputs){const x=inputs[0]?.[0];if(!x?.length)return true;const f0=currentFrame;
 if(this.pos<f0-1)this.pos=f0;
 for(;;){const p=Math.floor(this.pos),i=p-f0;if(i+1>=x.length)break;
 const a=i<0?this.prev:x[i],f=this.pos-p;if(!this.n)this.t0=this.pos/sampleRate;
 this.out[this.n++]=a+(x[i+1]-a)*f;
 if(this.n===320){this.port.postMessage({t:this.t0,x:this.out.slice()});this.n=0;}this.pos+=this.step;}
 this.prev=x[x.length-1];return true;}}
registerProcessor('robo-mic',Mic16k);`;

export class RoboClient {
 constructor(on={}){this.on=on;this.ctx=null;this.ws=null;this.stream=null;this.sources=[];this.next=0;this.queued={};this.cancelledReplies=new Set();this.epoch=0;this.connectionEpoch=0;this.startPromise=null;this.metrics={mic_blocks:0,tagged_blocks:0,chunks:0,reply_chunks:0};}
 async start(withMic=true){
  if(this.startPromise){await this.startPromise;if(withMic&&!this.stream)return this.start(true);return;}
  const promise=this._start(withMic,this.connectionEpoch);this.startPromise=promise;
  try{return await promise;}finally{if(this.startPromise===promise)this.startPromise=null;}
 }
 async _start(withMic,connectionEpoch){
  try{
  if(!this.ctx){this.ctx=new AudioContext({latencyHint:'interactive'});this.clock=new PlaybackClock(this.ctx);
   this.output=this.ctx.createGain();this.analyser=this.ctx.createAnalyser();this.analyser.fftSize=256;
   this.output.connect(this.analyser);this.analyser.connect(this.ctx.destination);}
  // A typed session may already own a context suspended by the browser. The
  // microphone button is a fresh user gesture: resume capture as well as TTS.
  if(this.ctx.state&&this.ctx.state!=='running'){
   let timer;try{await Promise.race([this.ctx.resume(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Click Enable microphone & sound to allow playback.')),600);})]);}finally{clearTimeout(timer);}
   if(this.ctx.state!=='running')throw new Error('Click Enable microphone & sound to allow playback.');
  }
  if(connectionEpoch!==this.connectionEpoch)throw new Error('Conversation start cancelled.');
  if(withMic&&!this.stream){let timer;try{await Promise.race([this.microphone(connectionEpoch),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Please allow microphone access, then enable Robby.')),20000);})]);}finally{clearTimeout(timer);}}
  if(this.ws?.readyState===1)return;
  await new Promise((resolve,reject)=>{const u=new URL('/ws',location.href);u.protocol=location.protocol==='https:'?'wss:':'ws:';
   // These counters describe THIS connection, like input_status on the server.
   this.sentMicSamples=0;this.captureClock=null;this.quickReplyId=null;this.metrics={mic_blocks:0,tagged_blocks:0,chunks:0,reply_chunks:0};
   const ws=this.ws=new WebSocket(u);ws.binaryType='arraybuffer';
   let opened=false;
   ws.onopen=()=>{if(connectionEpoch!==this.connectionEpoch){ws.close();reject(new Error('Conversation start cancelled.'));return;}opened=true;this.on.connection?.(true);resolve();};
   ws.onerror=()=>reject(new Error('Robo is busy or disconnected. End another session or try again shortly.'));
   ws.onclose=()=>{if(!opened)reject(new Error('Conversation start cancelled or server busy.'));if(this.ws===ws){this.ws=null;this.stop();}};
   ws.onmessage=e=>{if(this.ws!==ws)return;if(e.data instanceof ArrayBuffer)this.play(e.data);else{const m=JSON.parse(e.data);if(m.type==='reply'&&m.sentence_role==='reaction')this.quickReplyId=m.reply_id;if(m.type==='reply_cancelled'&&m.reply_id)this.cancelledReplies.add(m.reply_id);if(m.type==='barge')this.stopAudio();this.on.message?.(m);}};});
  }catch(error){if(connectionEpoch===this.connectionEpoch)this.stop();throw error;}
 }
 async microphone(connectionEpoch){
  const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
  if(connectionEpoch!==this.connectionEpoch){stream.getTracks().forEach(t=>t.stop());throw new Error('Microphone start cancelled.');}
  this.stream=stream;
  const latency=this.stream.getAudioTracks()[0].getSettings().latency;
  this.clock.inputLatency=Number.isFinite(latency)?latency:null;
  const url=URL.createObjectURL(new Blob([WORKLET],{type:'application/javascript'}));
  try{await this.ctx.audioWorklet.addModule(url);}finally{URL.revokeObjectURL(url);}
  if(connectionEpoch!==this.connectionEpoch){stream.getTracks().forEach(t=>t.stop());throw new Error('Microphone start cancelled.');}
  this.micSource=this.ctx.createMediaStreamSource(this.stream);this.worklet=new AudioWorkletNode(this.ctx,'robo-mic');
  const silent=this.ctx.createGain();silent.gain.value=0;this.micSource.connect(this.worklet);this.worklet.connect(silent);silent.connect(this.ctx.destination);
  this.captureStarted=performance.now();this.lastCapture=0;this.captureBlocks=0;
  this.stream.getAudioTracks()[0].onended=()=>{if(connectionEpoch===this.connectionEpoch){this.on.input?.({state:'ended'});this.stop();}};
  this.worklet.port.onmessage=e=>{if(connectionEpoch!==this.connectionEpoch)return;
   const x=e.data.x;this.captureBlocks++;this.lastCapture=performance.now();
   if(!this.lastInputNotice||this.lastCapture-this.lastInputNotice>250){
    this.lastInputNotice=this.lastCapture;let square=0;for(const v of x)square+=v*v;
    this.on.input?.({state:this.ctx.state,rms:Math.sqrt(square/x.length),captured:this.captureBlocks,sent:this.metrics.mic_blocks});
   }this.mic(e.data.t,x);};this.on.microphone?.(true);
 }
 mic(t,x){
  if(this.ws?.readyState!==1)return;
  // Map video performance timestamps onto the same received-sample timeline.
  // Input-device latency is independent of the agent playback/echo clock.
  this.captureClock={audio:(this.sentMicSamples||0)/16000,perf:performance.now()+(t-this.ctx.currentTime)*1000-(this.clock.inputLatency||0)*1000};
  this.sentMicSamples=(this.sentMicSamples||0)+x.length;
  // Split at scheduled playback edges instead of mislabeling the remainder
  // of a 20-ms block when playback begins, stops or has a buffered gap.
  for(const part of this.clock.pieces(t,x.length)){
   const buffer=new ArrayBuffer(8+2*(part.end-part.first)),dv=new DataView(buffer),pcm=new Int16Array(buffer,8);
   dv.setInt32(0,part.id,true);dv.setInt32(4,part.at,true);
   for(let i=part.first;i<part.end;i++)pcm[i-part.first]=Math.max(-32768,Math.min(32767,Math.round(x[i]*32767)));
   this.ws.send(buffer);this.metrics.mic_blocks++;if(part.at>=0)this.metrics.tagged_blocks++;
  }
  if(this.metrics.mic_blocks%25===0)this.on.sync?.(this.clock.report());
 }
 play(raw){
  const id=new DataView(raw).getInt32(0,true);
  if(id&&this.cancelledReplies.has(id))return; // In-flight chunks cannot restart a stopped reply.
  const pcm=new Int16Array(raw,4),b=this.ctx.createBuffer(1,pcm.length,24000),ch=b.getChannelData(0);
  if(id&&this.sources.some(s=>s.roboId===0)){
   for(const s of this.sources.filter(s=>s.roboId===0)){s.onended=null;try{s.stop();}catch{}}
   this.sources=this.sources.filter(s=>s.roboId!==0);if(!this.sources.length)this.next=0;
  }
  for(let i=0;i<pcm.length;i++)ch[i]=pcm[i]/32768;
  const source=this.ctx.createBufferSource();source.roboId=id;source.buffer=b;source.connect(this.output);
  const firstReply=!!id&&!this.queued[id];
  this.startupReserve??=.18;
  if(id&&!firstReply&&this.next>0){const gap=Math.max(0,this.ctx.currentTime-this.next);
   if(gap>.02){this.startupReserve=Math.min(.65,Math.max(this.startupReserve,gap+.12));this.metrics.underflows=(this.metrics.underflows||0)+1;this.metrics.max_gap_ms=Math.max(this.metrics.max_gap_ms||0,gap*1000);}}
  // The short optional reaction has 160-ms TTS chunks. Keep adaptive recovery
  // after an underrun. A 220-ms reserve covers admission of the second TTS
  // prefix while keeping warm first playback below the old 300-ms floor.
  const reserve=id===this.quickReplyId
   ? Math.max(.22,this.startupReserve)
   : Math.max(this.startupReserve,b.duration<.8?.3:.18);
  if(firstReply)this.metrics.startup_buffer_ms=reserve*1000;
  const start=playbackStart(this.ctx,this.next,firstReply,reserve),end=start+b.duration;
  const first=this.queued[id]||0;this.queued[id]=first+pcm.length;
  if(id)this.clock.add(id,start,pcm.length,first);
  const epoch=this.epoch;this.next=end;source.start(start);this.sources.push(source);this.metrics.chunks++;if(id)this.metrics.reply_chunks++;
  this.on.audio?.({reply_id:id,first:firstReply,playback_lead_ms:Math.max(0,start-this.ctx.currentTime)*1000});
  this.on.playing?.(true);
  source.onended=()=>{this.sources=this.sources.filter(s=>s!==source);
   if(epoch!==this.epoch)return;
   if(!id){if(!this.sources.length)this.on.playing?.(false);return;}
   const complete=()=>{if(epoch!==this.epoch||this.next>end+1e-6)return;
    if(this.clock.outputHead()+1e-4<end){setTimeout(complete,10);return;}
    this.send('played:'+id+':'+this.queued[id]);this.on.playing?.(false);};complete();};
 }
 send(text){if(this.ws?.readyState===1)this.ws.send(text);}
 audioTimeAt(at=performance.now()){return this.captureClock?Math.max(0,this.captureClock.audio+(at-this.captureClock.perf)/1000):null;}
 stopAudio(){this.epoch++;for(const id of Object.keys(this.queued))if(+id)this.cancelledReplies.add(+id);for(const s of this.sources){s.onended=null;try{s.stop();}catch{}}
  this.sources=[];this.next=0;this.clock?.stop();this.on.playing?.(false);}
 stop(){this.connectionEpoch++;this.startPromise=null;this.stopAudio();this.sentMicSamples=0;this.captureClock=null;const ws=this.ws;this.ws=null;ws?.close();this.stream?.getTracks().forEach(t=>t.stop());this.stream=null;
  this.micSource?.disconnect();this.worklet?.disconnect();this.ctx?.close().catch(()=>{});this.ctx=null;this.queued={};this.cancelledReplies.clear();this.on.microphone?.(false);this.on.connection?.(false);}
 level(){if(!this.analyser||this.ctx?.state!=='running')return 0;const x=new Float32Array(256);this.analyser.getFloatTimeDomainData(x);return Math.sqrt(x.reduce((a,b)=>a+b*b,0)/x.length);}
}
