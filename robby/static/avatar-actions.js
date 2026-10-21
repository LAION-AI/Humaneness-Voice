// Local, bounded visual performances. No network, hardware, or new models.
import {pose,mix,profiles,clamp} from './emotions.js';
const kinds=new Set(['emotion','nod','shake','spin','eye_roll','grimace','wink','curious']);
const slugs=new Set(profiles.map(p=>p.slug));
const ease=t=>t*t*t*(10+t*(-15+6*t));
export function validateSteps(steps){
 if(!Array.isArray(steps)||!steps.length||steps.length>6)return [];
 let total=0,last=0;const out=[];
 for(const s of steps){
  if(!s||!kinds.has(s.kind)||!Number.isFinite(s.duration_s)||s.duration_s<1||s.duration_s>6)return [];
  const sentence=s.sentence??last;if(!Number.isInteger(sentence)||sentence<last||sentence>11)return [];
  if(s.kind==='emotion'&&!slugs.has(s.emotion))return [];
  if(s.kind==='spin'&&!['clockwise','counterclockwise'].includes(s.direction))return [];
  if(s.kind==='grimace'&&!['crossed','opposite','wiggle','cheeky'].includes(s.variant))return [];
  const duration_s=s.kind==='spin'?Math.max(3,s.duration_s):s.duration_s;
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
 if(s.kind==='curious'){target=pose('interest',.7,0);if(!reduced)motion.tilt=-5*weight;}
 if(s.kind==='wink'){target=pose('amusement',.7,0);controls.blinkLeft=.92*weight;eyes=true;}
 if(s.kind==='eye_roll'){
  eyes=true;const a=2*Math.PI*ease(p);controls.eyeLeftX=controls.eyeRightX=(reduced?0:Math.sin(a)*.7*weight);
  controls.eyeLeftY=controls.eyeRightY=(reduced?-.4:-Math.cos(a)*.7)*weight;
  target=pose('teasing',.45,0);
 }
 if(s.kind==='grimace'){
  eyes=true;target=pose('amusement',.75,0);target.actions.smile=.7;target.actions.brow_arch=.4;
  if(s.variant==='crossed'){controls.eyeLeftX=.8*weight;controls.eyeRightX=-.8*weight;}
  if(s.variant==='opposite'){controls.eyeLeftX=-.65*weight;controls.eyeRightX=.65*weight;controls.eyeLeftY=-.45*weight;controls.eyeRightY=.45*weight;}
  if(s.variant==='wiggle'){const w=reduced?.5:Math.sin(p*Math.PI*4);controls.eyeLeftX=w*.65*weight;controls.eyeRightX=-w*.65*weight;target.actions.brow_asym=.3*w;}
  if(s.variant==='cheeky'){controls.blinkRight=.8*weight;target.actions.mouth_asym=.35;controls.eyeLeftX=-.4*weight;}
 }
 // Grimaces never enlarge the pupils, bare teeth, stretch the mouth or snap.
 return {pose:mix(base,target,weight),controls,motion,eyes,weight,kind:s.kind,progress:p};
}
export class AvatarActions{
 constructor(){this.enabled=true;this.idleEnabled=true;this.listeningEnabled=true;this.queue=[];this.plans=new Map();this.started=new Set();this.lastActivity=performance.now();this.nextIdle=this.lastActivity+45000;this.lastListening=-Infinity;this.active=null;this.idleIndex=0;}
 clear(){this.queue=[];this.plans.clear();this.started.clear();this.active=null;this.touch();}
 touch(now=performance.now()){this.lastActivity=now;this.nextIdle=Math.max(this.nextIdle,now+45000);if(this.queue[0]?.source==='idle')this.queue=[];}
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
   const gentle=['curious','wink'];this.play([{kind:gentle[this.idleIndex++%gentle.length],duration_s:2}],'idle',now);this.nextIdle=now+60000;
  }
  const q=this.queue[0];this.active=q?{kind:q.step.kind,source:q.source}:null;
  return q?sampleAction(q.step,(now-q.start)/(q.end-q.start),base,reduced):{pose:base,controls:{},motion:{},eyes:false};
 }
}
