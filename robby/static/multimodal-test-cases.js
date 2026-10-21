// Deterministic behavioral fixtures. Not used by production UI.
import {VisualSpeakers,People,PROFILE_KEY} from './visual-speakers.js';
const check=(b,m)=>{if(!b)throw Error(m);};
export function testFace(x=.3,open=0,scale=1){const p=Array.from({length:478},()=>({x,y:.45,z:0})),put=(i,dx,dy)=>p[i]={x:x+dx*scale,y:.45+dy*scale,z:0};
 put(10,0,-.17);put(152,0,.2);put(234,-.1,0);put(454,.1,0);put(33,-.065,-.06);put(263,.065,-.06);put(133,-.035,-.06);put(362,.035,-.06);put(1,0,0);put(61,-.05,.075);put(291,.05,.075);put(13,0,.075-open*.05);put(14,0,.075+open*.05);return p;}
export function behaviorChecks(){
 const v=new VisualSpeakers();let id;
 for(let i=0;i<20;i++){const faces=i%2?[testFace(.7,.6),testFace(.3,.6)]:[testFace(.3,.6),testFace(.7,.6)],t=v.update(faces,i*100);if(!id)id=t[0].id;check(t.every(t=>!t.speaking),'held open mouth was speech');}
 check(v.tracks.size===2,'unstable/reordered IDs');check([...v.tracks.values()].find(t=>t.box.x<.3).id===id,'reordered face identity changed');
 const motion=new VisualSpeakers();let detected=0;const start=performance.now();
 for(let i=0;i<1000;i++){const t=motion.update([testFace(.3,i%2?.30:.01),testFace(.7,0)],i*100);if(t.find(t=>t.box.x<.3).speaking)detected++;check(!t.find(t=>t.box.x>.5).speaking,'static closed mouth false positive');}
 const geometryMs=(performance.now()-start)/1000;check(detected>950,'moving mouth not detected');
 const modest=new VisualSpeakers();let modestDetected=0,firstModest=null;
 for(let i=0;i<40;i++){const t=modest.update([testFace(.3,.10+.03*Math.sin(i*.1*2*Math.PI*2))],i*100)[0];if(t.speaking){modestDetected++;firstModest??=i*100;}}
 check(modestDetected>=30,'modest natural-rate articulation was blocked');check(firstModest<=800,'modest articulation was detected too late');
 const jitter=new VisualSpeakers(),held=new VisualSpeakers(),slow=new VisualSpeakers();
 for(let i=0;i<40;i++){
  check(!jitter.update([testFace(.3,.2+.006*Math.sin(i*2))],i*100)[0].speaking,'landmark jitter became speech');
  check(!held.update([testFace(.3,i<10?.02:.4)],i*100)[0].speaking,'opening once and holding became speech');
  check(!slow.update([testFace(.3,.12+.03*Math.sin(i*.1*.4*Math.PI))],i*100)[0].speaking,'slow non-speech motion became speech');
 }
 const before=[...motion.tracks.keys()];motion.update([],100000);motion.update([testFace(.3,0)],100050);check(before.includes([...motion.tracks.keys()][0]),'short dropout lost ID');motion.update([],102000);check(motion.tracks.size===0,'expired faces retained');
 const rigid=new VisualSpeakers();for(let i=0;i<30;i++){const t=rigid.update([testFace(.3+i*.001,.2,1+i*.005)],i*100);check(t.every(t=>!t.speaking),'rigid motion became speech');}
 const overlap=new VisualSpeakers();for(let i=0;i<20;i++)overlap.update([testFace(.3,i%2?.3:.01),testFace(.7,i%2?.3:.01)],i*100);
 check([...overlap.tracks.values()].every(t=>t.speaking),'two simultaneous moving mouths not detected');check(!overlap.active,'ambiguous overlap selected a speaker');
 const storage=new Map(),adapter={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},people=new People(adapter),d=Array(128).fill(0);d[0]=1;
 people.bind('person-a','Alice',d);check(!storage.has(PROFILE_KEY),'biometrics persisted without consent');people.setConsent(true);check(new People(adapter).match(d)?.name==='Alice','consented face not restored');people.setConsent(false);check(!storage.has(PROFILE_KEY),'consent withdrawal retained profiles');
 const wrong=Array(128).fill(0);wrong[1]=1;check(!people.match(wrong),'unknown face forced identity');people.clear();check(!people.profiles.size,'forget failed');
 return {two_faces_stable:true,static_open_rejected:true,static_closed_rejected:true,rigid_motion_rejected:true,modest_articulation_frames:modestDetected,modest_onset_ms:firstModest,jitter_rejected:true,held_open_transition_rejected:true,slow_motion_rejected:true,moving_frames_detected:detected,geometry_two_faces_ms:geometryMs,consent_checks:true};
}
function request(worker,data,transfer=[]){return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Worker timed out')),90000);worker.onmessage=e=>{clearTimeout(timer);e.data.error?reject(Error(e.data.error)):resolve(e.data);};worker.onerror=e=>{clearTimeout(timer);reject(Error(e.message));};worker.postMessage(data,transfer);});}
export async function realModels(){
 const image=new Image();image.crossOrigin='anonymous';image.src='https://raw.githubusercontent.com/scikit-image/scikit-image/v0.19.3/skimage/data/astronaut.png';await image.decode();
 const canvas=document.createElement('canvas');canvas.width=768;canvas.height=384;const c=canvas.getContext('2d');c.drawImage(image,100,0,300,300,0,0,384,384);c.drawImage(image,100,0,300,300,384,0,384,384);
 const worker=new Worker('/static/face-camera-worker.js');await request(worker,{type:'load'});let result;const timings=[];
 try{for(let i=0;i<6;i++){const frame=await createImageBitmap(canvas);result=await request(worker,{type:'frame',at:performance.now(),frame},[frame]);timings.push(result.ms);}}finally{worker.terminate();}
 check(result.result.faceLandmarks.length>=2,'Real model detected '+result.result.faceLandmarks.length+' instead of two public test faces');
 const points=result.result.faceLandmarks[0],l=points[33],r=points[263],crop=document.createElement('canvas');crop.width=crop.height=150;const ctx=crop.getContext('2d');
 ctx.translate(75,55);ctx.scale(60/Math.hypot((r.x-l.x)*768,(r.y-l.y)*384),60/Math.hypot((r.x-l.x)*768,(r.y-l.y)*384));ctx.rotate(-Math.atan2((r.y-l.y)*384,(r.x-l.x)*768));ctx.drawImage(canvas,-(l.x+r.x)*768/2,-(l.y+r.y)*384/2);
 const data=ctx.getImageData(0,0,150,150).data,pixels=new Uint8Array(67500);for(let i=0,j=0;i<data.length;i+=4){pixels[j++]=data[i];pixels[j++]=data[i+1];pixels[j++]=data[i+2];}
 const idworker=new Worker('/static/face-id-worker.js'),results=[];let last;
 try{for(let i=0;i<3;i++){const copy=pixels.slice(),value=await request(idworker,{id:'test',pixels:copy.buffer,generation:0},[copy.buffer]);check(value.face.length===128&&value.face.every(Number.isFinite),'Invalid real embedding');if(last)check(Math.hypot(...value.face.map((v,k)=>v-last[k]))<.001,'Non-repeatable descriptor');last=value.face;results.push({ms:value.ms,backend:value.backend});}}finally{idworker.terminate();}
 return {detected_faces:result.result.faceLandmarks.length,landmark_ms:timings,embedding:results,embedding_dimensions:128};
}
