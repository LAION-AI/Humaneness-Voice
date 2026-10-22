// Local, bounded visual performances. No network, hardware, or new models.
import {pose,mix,profiles,clamp} from './emotions.js';
const kinds=new Set(['emotion','nod','shake','spin','carousel','pet','eye_roll','grimace','wink','curious']);
export const GRIMACES=['crossed','opposite','wiggle','cheeky','goofy','squishy','fish','peekaboo'];
const slugs=new Set(profiles.map(p=>p.slug));
const ease=t=>t*t*t*(10+t*(-15+6*t));
export function validateSteps(steps){
 if(!Array.isArray(steps)||!steps.length||steps.length>6)return [];
 let total=0,last=0;const out=[];
 for(const s of steps){
  if(!s||!kinds.has(s.kind)||!Number.isFinite(s.duration_s)||s.duration_s<1||s.duration_s>6)return [];
  const sentence=s.sentence??last;if(!Number.isInteger(sentence)||sentence<last||sentence>11)return [];
  if(s.kind==='emotion'&&!slugs.has(s.emotion))return [];
  if(['spin','carousel'].includes(s.kind)&&!['clockwise','counterclockwise'].includes(s.direction))return [];
  if(s.kind==='grimace'&&!GRIMACES.includes(s.variant))return [];
  const duration_s=['spin','carousel'].includes(s.kind)?Math.max(3,s.duration_s):s.duration_s;
  total+=duration_s;last=sentence;out.push({...s,duration_s,sentence});
 }
 return total<=24?out:[];
}
export function sampleAction(s,progress,base=pose(),reduced=false){
 const p=clamp(progress,0,1),weight=ease(Math.min(1,p*5))*ease(Math.min(1,(1-p)*5));
 const wave=Math.sin(p*Math.PI*4)*weight,controls={},motion={pan:0,tilt:0,spin:0};
 let target=base,eyes=false;
 if(s.kind==='emotion'){target=pose(s.emotion,1,0);eyes=true;}
 if(s.kind==='nod'&&!reduced)motion.tilt=9*wave;
 if(s.kind==='shake'&&!reduced)motion.pan=17*wave;
 // +Y is up; positive yaw is counterclockwise when viewed from above.
 if(s.kind==='spin'&&!reduced)motion.spin=(s.direction==='counterclockwise'?1:-1)*360*ease(p);
 if(s.kind==='carousel'){
  target=pose('joy',1,0);target.actions.mouth_open=.7;eyes=true;
  if(!reduced)motion.spin=(s.direction==='counterclockwise'?1:-1)*720*ease(p);
 }
 if(s.kind==='pet'){
  target=pose('contentment',1,0);eyes=true;
  Object.assign(target.actions,{eye_open:.22,squint:.35,lid_curve:.8,brow_arch:.55,
   smile:.95,mouth_open:0,mouth_width:1.1,cheek_lift:.65});target.blush=.7;
  if(!reduced)motion.tilt=(-4+Math.sin(p*Math.PI*2))*weight;
 }
 if(s.kind==='curious'){target=pose('interest',.7,0);if(!reduced)motion.tilt=-5*weight;}
 if(s.kind==='wink'){target=pose('amusement',.7,0);controls.blinkLeft=.92*weight;eyes=true;}
 if(s.kind==='eye_roll'){
  eyes=true;const a=2*Math.PI*ease(p);controls.eyeLeftX=controls.eyeRightX=(reduced?0:Math.sin(a)*.7*weight);
  controls.eyeLeftY=controls.eyeRightY=(reduced?-.4:-Math.cos(a)*.7)*weight;
  target=pose('teasing',.45,0);
 }
 if(s.kind==='grimace'){
  // Exaggeration is action-only: normal emotions, idle antics and gaze keep
  // their original geometry. Slow squash/stretch, never strobing or jitter.
  eyes=true;target=pose('amusement',.9,0);controls.cartoon=weight;
  Object.assign(target.actions,{smile:1.3,brow_arch:1,eye_open:1.4,
   squint:0,pupil_scale:.68,mouth_width:1.65,mouth_open:.65,cheek_lift:.8});
  target.blush=.75;target.teeth=0;target.tongue=.85;
  const w=reduced?.55:Math.sin(p*Math.PI*4);
  if(s.variant==='crossed'){
   controls.eyeLeftX=weight;controls.eyeRightX=-weight;
   controls.eyeLeftY=controls.eyeRightY=.35*weight;
   Object.assign(target.actions,{brow_raise:1.1,eye_asym:.25,mouth_asym:.7});
  }
  if(s.variant==='opposite'){
   controls.eyeLeftX=-.85*weight;controls.eyeRightX=.85*weight;
   controls.eyeLeftY=-.8*weight;controls.eyeRightY=.8*weight;
   Object.assign(target.actions,{eye_asym:.7,brow_asym:1.4,mouth_asym:-1.2,mouth_open:.85});
  }
  if(s.variant==='wiggle'){
   controls.eyeLeftX=w*weight;controls.eyeRightX=-w*weight;
   controls.eyeLeftY=-.55*w*weight;controls.eyeRightY=.55*w*weight;
   Object.assign(target.actions,{eye_asym:.65*w,brow_asym:1.5*w,
    mouth_asym:1.1*w,mouth_open:.7+.15*w});
  }
  if(s.variant==='cheeky'){
   controls.blinkRight=.96*weight;controls.eyeLeftX=-.85*weight;
   controls.eyeLeftY=-.4*weight;
   Object.assign(target.actions,{brow_asym:1.3,brow_raise:.5,mouth_asym:1.4,mouth_open:.9});
  }
  if(s.variant==='goofy'){
   controls.eyeLeftX=.9*weight;controls.eyeRightX=-.7*weight;
   controls.eyeLeftY=-.8*weight;controls.eyeRightY=-.6*weight;
   Object.assign(target.actions,{eye_asym:-.85,brow_asym:-1.6,mouth_asym:1.5,mouth_open:.95});
  }
  if(s.variant==='squishy'){
   Object.assign(target.actions,{eye_open:.25,squint:.4,brow_raise:-.4,
    mouth_width:1.7,mouth_open:.45,smile:1.6,cheek_lift:1});
  }
  if(s.variant==='fish'){
   controls.eyeLeftX=weight;controls.eyeRightX=-weight;
   Object.assign(target.actions,{eye_open:1.6,brow_raise:1.2,mouth_width:.4,
    mouth_open:.85,smile:.1,mouth_asym:0});target.tongue=0;
  }
  if(s.variant==='peekaboo'){
   controls.blinkLeft=(reduced?.9:Math.max(0,w)*.96)*weight;
   controls.blinkRight=(reduced?0:Math.max(0,-w)*.96)*weight;
   controls.eyeLeftX=.6*weight;controls.eyeRightX=-.6*weight;
   Object.assign(target.actions,{brow_asym:1.4*w,mouth_asym:-w,mouth_open:.8});
  }
  if(!reduced){motion.pan=5*w*weight;motion.tilt=-3*weight;}
 }
 // Smooth envelope restores the incoming expression at both endpoints.
 return {pose:mix(base,target,weight),controls,motion,eyes,weight,kind:s.kind,progress:p};
}
export class AvatarActions{
 constructor(){this.enabled=true;this.idleEnabled=true;this.pettingEnabled=true;this.petSpeechEnabled=true;this.petInvitesEnabled=true;this.listeningEnabled=true;this.queue=[];this.plans=new Map();this.started=new Set();this.lastActivity=performance.now();this.nextIdle=this.lastActivity+60000;this.lastListening=-Infinity;this.active=null;this.idleIndex=0;this.onIdle=null;}
 clear(){this.queue=[];this.plans.clear();this.started.clear();this.active=null;this.touch();}
 touch(now=performance.now()){this.lastActivity=now;this.nextIdle=Math.max(this.nextIdle,now+60000);if(this.queue[0]?.source==='idle')this.queue=[];}
 set(id,steps,plan){const safe=validateSteps(steps);if(!safe.length||!id)return;this.plans.set(id,{steps:safe,plan});while(this.plans.size>20)this.plans.delete(this.plans.keys().next().value);}
 play(steps,source='preview',now=performance.now()){
  if(!this.enabled)return false;const safe=validateSteps(steps);if(!safe.length)return false;
  if(source==='preview'){this.clear();this.touch(now);}
  if(this.queue.length+safe.length>6)return false;
  let at=Math.max(now,this.queue.at(-1)?.end??now);
  for(const step of safe){this.queue.push({step,start:at,end:at+step.duration_s*1000,source});at+=step.duration_s*1000;}
  return true;
 }
 listen(kind,now=performance.now()){
  if(!this.enabled||!this.listeningEnabled||this.queue.length||now-this.lastListening<10000||!['nod','shake'].includes(kind))return false;
  this.lastListening=now;return this.play([{kind,duration_s:1.6}],'listening',now);
 }
 frame(now,base,{position=null,busy=false,reduced=false,hidden=false,calibrating=false}={}){
  if(!this.enabled||hidden||calibrating){this.clear();return {pose:base,controls:{},motion:{},eyes:false};}
  if(busy)this.touch(now);
  if(position&&this.plans.has(position.id)){
   const {steps,plan}=this.plans.get(position.id),seconds=position.sample/24000;
   steps.forEach((s,i)=>{const key=position.id+':'+i;
    const cues=plan?.sentences||[],at=cues[Math.min(s.sentence,Math.max(0,cues.length-1))]?.start_s??0;
    if(!this.started.has(key)&&seconds>=at){this.started.add(key);this.play([s],'reply',now);}
   });
   while(this.started.size>120)this.started.delete(this.started.values().next().value);
  }
  while(this.queue.length&&now>=this.queue[0].end)this.queue.shift();
  if(!this.queue.length&&!busy&&!reduced&&this.idleEnabled&&now>=this.nextIdle){
   const i=this.idleIndex++,variant=GRIMACES[i%GRIMACES.length];
   const steps=[{kind:'grimace',variant,duration_s:2.6},
    ...(i%2===0?[{kind:'carousel',direction:i%4?'counterclockwise':'clockwise',duration_s:4.5}]:[]),
    {kind:'grimace',variant:GRIMACES[(i+3)%GRIMACES.length],duration_s:2.4}];
   this.play(steps,'idle',now);this.nextIdle=now+60000;this.onIdle?.({variant,carousel:i%2===0});
  }
  const q=this.queue[0];this.active=q?{kind:q.step.kind,source:q.source}:null;
  return q?sampleAction(q.step,(now-q.start)/(q.end-q.start),base,reduced):{pose:base,controls:{},motion:{},eyes:false};
 }
}
