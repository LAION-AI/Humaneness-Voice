// Eye-position following, not facial-pose imitation. No camera frames leave here.
// Pinhole/IPD depth is a heuristic, not a metric depth sensor or eye-gaze detector.
const limit=(v,a,b)=>Math.max(a,Math.min(b,v));
const rad=d=>d*Math.PI/180,deg=r=>r*180/Math.PI;
export const GAZE_KEY='robby.eyeContact.v1';
export const PRIOR=[[0,.35,0,0],[0,0,.25,0]];
const mean=(a,b)=>({x:(a.x+b.x)/2,y:(a.y+b.y)/2,z:((a.z||0)+(b.z||0))/2});
export function eyeObservation(result,width=640,height=480){
 const p=result?.faceLandmarks?.[0];
 if(!p||![33,133,362,263].every(i=>p[i]&&Number.isFinite(p[i].x)&&Number.isFinite(p[i].y)))return null;
 const left=mean(p[33],p[133]),right=mean(p[362],p[263]),center=mean(left,right);
 const dx=left.x-right.x,dy=(left.y-right.y)*height/width,dz=left.z-right.z;
 // z has x's scale in MediaPipe's weak-perspective landmarks. Use the
 // 3-D eye baseline, not its horizontal projection: roll/yaw alone should
 // not look like moving farther away. Limit uncertain depth corrections.
 const projected=Math.hypot(dx,dy),baseline=Math.hypot(dx,dy,limit(dz,-projected,projected));
 if(projected<.018||baseline>.65||center.x<0||center.x>1||center.y<0||center.y>1)return null;
 const fx=1/(2*Math.tan(rad(65)/2)),fy=fx*width/height;
 const z=limit(fx*.063/baseline,.22,2.8);
 return {u:center.x,v:center.y,eyeSpan:baseline,depth:z,
  rayX:(.5-center.x)/fx,rayY:(.5-center.y)/fy};
}
export function gazeFeatures(o){
 // Modest virtual pivot setback: an explicit visual heuristic, not a
 // measured webcam/head dimension. Calibration can learn residual offsets.
 const gain=o.depth/(o.depth+.08);
 return [1,o.rayX*gain/.35,o.rayY*gain/.25,(1/o.depth-1/.65)*.35];
}
export function predictGaze(o,coefficients=PRIOR){
 const f=gazeFeatures(o),value=row=>deg(Math.atan(row.reduce((sum,v,i)=>sum+v*f[i],0)));
 return {pan:limit(value(coefficients[0]),-65,65),tilt:limit(value(coefficients[1]),-15,15),depth:o.depth};
}
function solve(matrix,vector){
 const a=matrix.map((row,i)=>[...row,vector[i]]);
 for(let k=0;k<4;k++){
  let pivot=k;for(let j=k+1;j<4;j++)if(Math.abs(a[j][k])>Math.abs(a[pivot][k]))pivot=j;
  [a[k],a[pivot]]=[a[pivot],a[k]];if(Math.abs(a[k][k])<1e-10)return null;
  const d=a[k][k];for(let i=k;i<=4;i++)a[k][i]/=d;
  for(let j=0;j<4;j++)if(j!==k){const d=a[j][k];for(let i=k;i<=4;i++)a[j][i]-=d*a[k][i];}
 }return a.map(row=>row[4]);
}
export function fitGaze(samples){
 const penalty=[.08,.15,.15,.6]; // strong depth prior until near/far examples exist
 return PRIOR.map((prior,axis)=>{
  const a=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>i===j?penalty[i]:0));
  const b=prior.map((v,i)=>v*penalty[i]);
  for(const sample of samples){
   const f=gazeFeatures(sample.observation),target=Math.tan(rad(limit(axis?sample.tilt:sample.pan,axis?-15:-65,axis?15:65)));
   for(let i=0;i<4;i++){b[i]+=f[i]*target;for(let j=0;j<4;j++)a[i][j]+=f[i]*f[j];}
  }
  const row=solve(a,b);return row&&row.every(Number.isFinite)?row:prior.slice();
 });
}
export class GazeTracker{
 constructor(storage){
  if(storage===undefined){try{storage=globalThis.localStorage;}catch{storage=null;}}
  this.storage=storage;this.samples=[];this.coefficients=PRIOR.map(r=>r.slice());this.filtered=null;this.latest=null;this.at=-Infinity;this.lastRead=-Infinity;
  try{const saved=JSON.parse(storage?.getItem(GAZE_KEY)||'null');
   if(saved?.version===1&&Array.isArray(saved.samples))this.samples=saved.samples.filter(s=>s&&Number.isFinite(s.pan)&&Number.isFinite(s.tilt)&&Math.abs(s.pan)<=65&&Math.abs(s.tilt)<=15&&s.observation&&['u','v','depth','rayX','rayY'].every(k=>Number.isFinite(s.observation[k]))&&s.observation.depth>=.22&&s.observation.depth<=2.8).slice(-128);
  }catch{/* Storage is optional, including private mode. */}
  this.coefficients=fitGaze(this.samples);
 }
 observe(result,width,height,now=performance.now()){
  const o=eyeObservation(result,width,height);if(!o)return false;
  const dt=limit(now-this.lastRead,0,250),alpha=this.latest&&now-this.at<750?1-Math.exp(-dt/130):1;
  const prev=this.latest;
  // A small position dead band suppresses landmark tremor, not deliberate motion.
  if(prev){for(const key of ['u','v','rayX','rayY','depth','eyeSpan']){
   const d=o[key]-prev[key];o[key]=prev[key]+alpha*(Math.abs(d)<(key==='depth'?.003:key==='eyeSpan'?.0003:.0007)?0:d);
  }}
  this.latest=o;this.lastRead=now;this.at=now;return true;
 }
 read(now=performance.now()){
  if(!this.latest||now-this.at>1300)return null;
  const target=predictGaze(this.latest,this.coefficients);
  // Hold through short dropouts, then gently return towards neutral.
  const weight=limit((1300-(now-this.at))/700,0,1);
  return {...target,pan:target.pan*weight,tilt:target.tilt*weight,confidence:weight};
 }
 accept(observation,pan,tilt){
  const sample={observation:{...observation},pan:limit(pan,-65,65),tilt:limit(tilt,-15,15)};
  const f=gazeFeatures(observation),same=this.samples.findIndex(s=>Math.hypot(...gazeFeatures(s.observation).map((v,i)=>v-f[i]))<.045);
  if(same>=0)this.samples[same]=sample;else this.samples.push(sample);
  this.samples=this.samples.slice(-128);this.refit();
 }
 refit(){this.coefficients=fitGaze(this.samples);try{this.storage?.setItem(GAZE_KEY,JSON.stringify({version:1,samples:this.samples}));}catch{}}
 undo(){this.samples.pop();this.refit();}
 clear(){this.samples=[];this.refit();}
 resetTracking(){this.latest=null;this.lastRead=this.at=-Infinity;}
}
