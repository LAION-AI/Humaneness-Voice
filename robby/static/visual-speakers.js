// Geometry and session identities only. No image upload, demographic inference,
// or identity decision from a static open mouth. Times are performance milliseconds.
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y,a.z&&b.z?a.z-b.z:0);
export function faceGeometry(points){
 if(!points||![13,14,61,291,33,263,1,152].every(i=>points[i]&&Number.isFinite(points[i].x)&&Number.isFinite(points[i].y)))return null;
 const xs=points.map(p=>p.x),ys=points.map(p=>p.y),x=Math.min(...xs),y=Math.min(...ys);
 const box={x,y,w:Math.max(...xs)-x,h:Math.max(...ys)-y};
 if(box.w<.045||box.h<.06)return null;
 const width=distance(points[61],points[291]),eyes=distance(points[33],points[263]);
 if(width<.015||eyes<.02)return null;
 return {box,open:distance(points[13],points[14])/width,
  jaw:distance(points[1],points[152])/eyes,width:width/eyes,points};
}
function iou(a,b){const area=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x))*Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return area/(a.w*a.h+b.w*b.h-area||1);}
function matchCost(a,b){const dx=(a.x+a.w/2)-(b.x+b.w/2),dy=(a.y+a.h/2)-(b.y+b.h/2);
 return Math.hypot(dx,dy)/Math.max(.08,a.w,b.w)+.35*Math.abs(Math.log(a.w/b.w))+.4*(1-iou(a,b));}
// Sample-rate-independent motion, not a mouth-open classifier. At 10 Hz the
// old per-frame threshold rejected modest articulation and required multiple
// complete syllables inside 700 ms. Ratios and a minimum excursion reject
// static mouths, landmark jitter and rigid translation/scale changes.
export function articulation(samples,key='open'){
 if(samples.length<5)return 0;
 const span=(samples.at(-1).at-samples[0].at)/1000;if(span<.35)return 0;
 let travel=0,turns=0,direction=0,extreme=samples[0][key],low=extreme,high=extreme;
 for(let i=1;i<samples.length;i++){
  const value=samples[i][key];travel+=Math.abs(value-samples[i-1][key]);low=Math.min(low,value);high=Math.max(high,value);
  // Count excursions from extrema, so several small movements can make one
  // opening/closing cycle without each individual frame exceeding a cutoff.
  if(!direction){if(Math.abs(value-extreme)>.012){direction=Math.sign(value-extreme);extreme=value;}}
  else if((value-extreme)*direction>=0)extreme=value;
  else if(Math.abs(value-extreme)>.012){turns++;direction=-direction;extreme=value;}
 }
 const range=high-low,minimum=key==='open'?.035:.05;
 if(!turns||range<minimum)return 0;
 return clamp((travel/span-.025)/.16)*clamp(range/(key==='open'?.055:.08));
}
export class VisualSpeakers{
 constructor(){this.tracks=new Map();this.next=0;this.active=null;this.changed=-Infinity;}
 clear(){this.tracks.clear();this.active=null;}
 update(faces,now,speakingEnabled=true){
  const geometries=faces.map(faceGeometry).filter(Boolean),old=[...this.tracks.values()].filter(t=>now-t.seen<1100);
  const pairs=[];for(let i=0;i<geometries.length;i++)for(const t of old)pairs.push({i,t,c:matchCost(t.box,geometries[i].box)});
  pairs.sort((a,b)=>a.c-b.c);const used=new Set(),assigned=new Map();
  for(const p of pairs){if(p.c>.95||used.has(p.t.id)||assigned.has(p.i))continue;
   // Crossing/overlapping boxes are unsafe identity evidence. Clear the ID
   // association rather than inheriting another person's face descriptor.
   const alternatives=pairs.filter(q=>q.i===p.i&&q.t.id!==p.t.id);
   if(alternatives.some(q=>Math.abs(q.c-p.c)<.12)){p.t.ambiguousUntil=now+800;p.t.identityEpoch=(p.t.identityEpoch||0)+1;p.t.personId=null;p.t.embedding=null;}
   assigned.set(p.i,p.t);used.add(p.t.id);
  }
  geometries.forEach((g,i)=>{
   let t=assigned.get(i);
   if(!t){t={id:'face-'+(++this.next),samples:[],confidence:0,speaking:false,seen:now,born:now,personId:null};this.tracks.set(t.id,t);}
   if(now-t.seen>350){t.samples=[];t.speaking=false;t.confidence=0;t.embedding=null;t.personId=null;t.identityEpoch=(t.identityEpoch||0)+1;t.ambiguousUntil=now+800;}
   t.seen=now;t.box=g.box;t.points=g.points;
   if(!speakingEnabled){t.samples=[];t.speaking=false;t.confidence=0;return;}
   t.samples.push({at:now,open:g.open,jaw:g.jaw,width:g.width});
   t.samples=t.samples.filter(s=>now-s.at<=800);
   // Mouth widening helps consonant-heavy/quiet speech. Jaw movement alone
   // is deliberately insufficient: nodding must not impersonate speaking.
   const raw=Math.max(articulation(t.samples),.85*articulation(t.samples,'width'));
   t.confidence=.45*t.confidence+.55*raw;
   if(now<(t.ambiguousUntil||0))t.confidence=0;
   t.speaking=t.speaking?t.confidence>.28:t.confidence>.55;
  });
  for(const [id,t] of this.tracks){if(now-t.seen>1100)this.tracks.delete(id);else if(now-t.seen>250){t.speaking=false;t.confidence=0;}}
  const visible=[...this.tracks.values()].filter(t=>now-t.seen<250);
  const talkers=visible.filter(t=>t.speaking).sort((a,b)=>b.confidence-a.confidence);
  const unambiguous=talkers.length===1||(talkers.length>1&&talkers[0].confidence-talkers[1].confidence>.25);
  const candidate=unambiguous?talkers[0]:null;
  if(candidate&&candidate.id!==this.active&&now-this.changed>600){this.active=candidate.id;this.changed=now;}
  if(!this.tracks.has(this.active)||now-(this.tracks.get(this.active)?.seen||0)>700)this.active=null;
  return visible;
 }
 snapshot(now){return [...this.tracks.values()].filter(t=>now-t.seen<250).map(t=>({id:t.id,confidence:t.confidence,speaking:t.speaking,ambiguous:now<(t.ambiguousUntil||0),person_id:t.personId||null}));}
 gazeFace(now){const active=this.tracks.get(this.active);if(active&&now-active.seen<500)return active;
  const visible=[...this.tracks.values()].filter(t=>now-t.seen<250);return visible.length===1?visible[0]:null;}
}
export const PROFILE_KEY='robby.consentedPeople.v1';
const validImage=value=>typeof value==='string'&&value.length<60000&&/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(value);
const l2=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
export function descriptor(value){if(!Array.isArray(value)||value.length!==128||!value.every(Number.isFinite)||Math.hypot(...value)<.05)throw Error('Invalid face descriptor');return value.slice();}
export class People{
 constructor(storage){this.storage=storage;this.profiles=new Map();this.consent=false;
  try{const d=JSON.parse(storage?.getItem(PROFILE_KEY)||'null');if(d?.consent===true&&d.version===1){this.consent=true;for(const p of (d.people||[]).slice(0,8))this.restore(p);}}catch{this.profiles.clear();this.consent=false;}}
 restore(p){const embedding=descriptor(p.face);if(!/^[a-zA-Z0-9-]{1,70}$/.test(p.id)||typeof p.name!=='string'||p.name.length>60)throw Error('Invalid person');
  this.profiles.set(p.id,{id:p.id,name:p.name,face:embedding,voice:p.voice||null,image:validImage(p.image)?p.image:null});}
 match(face){descriptor(face);const matches=[...this.profiles.values()].map(p=>({person:p,d:l2(face,p.face)})).sort((a,b)=>a.d-b.d);
  return matches[0]&&matches[0].d<.48&&(!matches[1]||matches[1].d-matches[0].d>.08)?matches[0].person:null;}
 bind(id,name,face,voice=null,image=null){descriptor(face);const existing=this.profiles.get(id);
  // Never replace a person's face from a conflicting descriptor.
  if(existing&&l2(existing.face,face)>.52)return false;
  this.profiles.set(id,{id,name,face:existing?.face||face.slice(),voice:voice||existing?.voice||null,image:existing?existing.image:(validImage(image)?image:null)});this.save();return true;}
 save(){this.storageError=null;try{if(this.consent){if(!this.storage)throw Error('Browser storage unavailable');this.storage.setItem(PROFILE_KEY,JSON.stringify({version:1,consent:true,people:[...this.profiles.values()].slice(0,8)}));}else this.storage?.removeItem(PROFILE_KEY);}catch(e){this.storageError=e.message;this.consent=false;}return !this.storageError;}
 setConsent(value){this.consent=!!value;this.save();}
 clear(){this.profiles.clear();this.consent=false;this.save();}
}
