// MIT. Robot transfer of cartoon-face-generator's EmoNet-Face 40 recipes.
// Intended authored expressions, not recognition or universal facial codes.
export const catalog=await fetch('/static/emotion-catalog.json').then(r=>r.json());
export const profiles=catalog.profiles;
export const clamp=(v,a,b)=>Math.max(a,Math.min(b,Number.isFinite(+v)?+v:0));
export const neutral={brow_raise:0,brow_inner:0,brow_outer:0,brow_arch:0,brow_asym:0,frown:0,
 eye_open:1,eye_asym:0,squint:0,lid_curve:0,lid_tilt:0,gaze_x:0,gaze_y:0,pupil_scale:1,
 mouth_width:1,smile:.15,mouth_open:0,mouth_asym:0,mouth_tension:0,mouth_y:0,cheek_lift:0,nose_lift:0,curl:0,warmth:.4};
// [head tilt, nod, lean, lift-left, lift-right, shrug, gaze sway, tempo]
const gesture={amusement:[-5,4,0,8,4,2,.1,1.6],elation:[-2,-3,0,25,25,0,.1,1.5],
 pleasure_ecstasy:[7,-3,0,1,1,0,0,.6],contentment:[2,1,0,0,0,0,0,.55],gratitude:[-5,5,0,4,4,1,0,.7],
 affection:[-8,2,-1,5,3,0,.1,.65],infatuation:[10,-2,2,5,8,3,.1,.6],hope:[-4,-5,0,7,9,0,0,1],
 triumph:[-3,-6,0,36,32,0,.1,1.3],pride:[0,-5,0,10,10,0,0,.7],interest:[-8,-1,3,5,0,0,.1,1],
 awe:[0,-7,-2,8,8,0,.1,.6],surprise:[-2,-5,-4,24,24,9,.1,1.5],concentration:[-2,2,4,0,0,0,0,.7],
 contemplation:[9,-3,0,1,17,0,.2,.6],relief:[-4,6,0,0,0,-3,0,.55],longing:[-10,2,-1,3,0,0,.1,.55],
 teasing:[-12,-1,2,18,2,3,.1,1.3],impatience:[6,2,0,6,0,2,.1,1.3],lust:[10,2,2,3,3,0,.1,.6],
 doubt:[12,-1,-2,1,16,5,.1,.8],fear:[-7,4,-5,15,13,12,.1,1.4],distress:[-9,5,-2,18,20,10,.15,1.2],
 confusion:[-12,1,0,14,6,8,.2,.8],embarrassment:[11,7,-2,2,8,6,.1,.7],shame:[-5,12,-4,0,0,6,0,.5],
 disappointment:[3,8,-3,0,0,-2,0,.6],sadness:[-8,10,-3,0,0,-1,0,.55],bitterness:[7,4,-1,4,0,1,0,.65],
 contempt:[-10,-4,-2,1,0,0,0,.7],disgust:[8,-1,-5,8,2,5,0,.8],anger:[0,4,5,12,14,4,0,1.1],
 malice:[-9,-3,3,10,4,0,.1,.8],sourness:[12,3,-3,2,5,7,0,.8],pain:[-10,7,-4,14,18,12,0,.8],
 helplessness:[-8,8,-3,26,26,14,0,.65],fatigue:[13,11,-4,0,0,-3,0,.4],numbness:[0,3,-2,0,0,0,0,.3],
 intoxication:[16,3,3,8,2,4,.3,.6],envy:[-6,0,-1,4,0,2,0,.8]};
function random(seed){let a=seed|0;return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
export function pose(slug='neutral',intensity=1,variation=.15,seed=42){
 const profile=profiles.find(p=>p.slug===slug),a={...neutral},amount=clamp(intensity,0,1);
 if(profile)for(const k of Object.keys(a))a[k]+=(Number(profile.actions[k]??neutral[k])-a[k])*amount;
 const g=(gesture[slug]||[0,0,0,0,0,0,0,.7]).map((x,i)=>i===7?x:x*amount);
 const r=random(seed),asym=(r()-.5)*clamp(variation,0,1)*amount,look=(r()-.5)*variation*amount;
 a.brow_asym+=asym*.18;a.mouth_asym+=asym*.13;a.gaze_x+=look*.2;g[0]+=asym*5;
 const options=profile?.options||{};
 return {slug,actions:a,gesture:g,teeth:options.mouth==='teeth'?amount:0,
  tongue:options.tongue&&options.tongue!=='none'?amount:0,blush:clamp(a.cheek_lift*.3+(['infatuation','embarrassment','lust'].includes(slug)?.5:0),0,1)};
}
export function mix(a,b,t){t=clamp(t,0,1);const actions={};for(const k of Object.keys(neutral))actions[k]=a.actions[k]*(1-t)+b.actions[k]*t;
 return {slug:t<.5?a.slug:b.slug,actions,gesture:a.gesture.map((v,i)=>v*(1-t)+b.gesture[i]*t),
 teeth:a.teeth*(1-t)+b.teeth*t,tongue:a.tongue*(1-t)+b.tongue*t,blush:a.blush*(1-t)+b.blush*t};}
export function cuePose(cue={},seed=42){return mix(pose(cue.emotion||'contentment',cue.intensity??.75,.22,seed),
 pose(cue.secondary||cue.emotion||'contentment',cue.intensity??.75,.22,seed+7),cue.blend??0);}
// DAC position, not message arrival. Cue durations are prompt estimates, NOT ASR alignment.
export class ActingTimeline{
 constructor(){this.plans=new Map();this.enabled=true;}
 set(id,plan){if(id&&plan?.sentences?.length){this.plans.set(id,plan);while(this.plans.size>20)this.plans.delete(this.plans.keys().next().value);}}
 at(position){if(!this.enabled||!position)return null;const p=this.plans.get(position.id);if(!p)return null;
  const cues=p.sentences,seconds=position.sample/24000;
  const i=cues.reduce((chosen,c,j)=>seconds>=c.start_s?j:chosen,0),cue=cues[i];
  return {...cue,pose:cuePose(cue,position.id*91+i),index:i,seconds,source:p.source,timing:p.timing};}
 clear(){this.plans.clear();}
}
