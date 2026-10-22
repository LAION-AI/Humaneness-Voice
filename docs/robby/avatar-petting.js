// Pointer-only affection: no camera, pressure sensor, recording or new library.
// The app owns speech; this component emits a bounded event after a real stroke.
import {sampleAction} from './avatar-actions.js';
import {mix} from './emotions.js';
const hand=`<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40"><path d="M12 32C8 28 4 22 6 20Q8 18 12 23L10 10Q10 6 13 6Q15 6 15 10L17 19L16 5Q16 2 19 3Q21 3 21 6L22 18L23 7Q24 4 26 6L27 20L29 12Q31 9 33 12L32 26Q31 34 26 36L17 36Z" fill="#fff0de" stroke="#775c54" stroke-width="1.7" stroke-linejoin="round"/></svg>`;
const cursor=`url("data:image/svg+xml,${encodeURIComponent(hand)}") 18 8, grab`;
export class StrokeTracker{
 constructor(){this.reset();}
 reset(){this.active=false;this.distance=0;this.moves=0;this.started=0;this.lastMove=0;this.last=null;}
 begin(x,y,now){this.reset();this.active=true;this.started=now;this.lastMove=now;this.last=[x,y];}
 move(x,y,now,inside){if(!this.active)return false;
  const d=Math.hypot(x-this.last[0],y-this.last[1]);this.last=[x,y];
  // Jumping across the whole screen is not a gentle stroke. Motion outside
  // the head contributes nothing, nor does holding the mouse stationary.
  if(inside&&d>=1&&d<=80){this.distance+=d;this.moves++;this.lastMove=now;return true;}return false;
 }
 sample(now){const value={duration_ms:Math.min(15000,Math.round(now-this.started)),distance:this.distance,moves:this.moves};
  const valid=this.active&&value.duration_ms>=450&&value.distance>=50&&value.moves>=5&&now-this.lastMove<1200;
  return valid?value:null;
 }
 finish(now){const result=this.sample(now);this.reset();return result;}
}
export function attachPetting({element,rig,actions,allowed=()=>true,onStart=()=>{},onPet=()=>{}}){
 const surface=element.querySelector('canvas,svg'),stroke=new StrokeTracker();
 let pointer=null,strength=0,lastMotion=-Infinity,lastTick=performance.now(),lastAttempt=-Infinity,lastReaction=-Infinity;
 function react(result,now){if(result&&now-lastAttempt>=1000&&now-lastReaction>=actions.petCooldownMs){lastAttempt=now;if(onPet(result)===true)lastReaction=now;}}
 function hit(x,y){const r=surface.getBoundingClientRect();
  if(rig.headRegion){const h=rig.headRegion();return ((x-r.left-h.x)/h.rx)**2+((y-r.top-h.y)/h.ry)**2<=1;}
  const b=surface.querySelector('#head')?.getBoundingClientRect();
  return b?((x-(b.left+b.right)/2)/(b.width/2))**2+((y-(b.top+b.bottom)/2)/(b.height/2))**2<=1:false;
 }
 const enabled=()=>actions.enabled&&actions.pettingEnabled&&allowed()&&!document.hidden;
 function reset(){stroke.reset();pointer=null;strength=0;lastMotion=-Infinity;surface.style.cursor='grab';}
 surface.addEventListener('pointerdown',e=>{
  if(e.button!==0||e.altKey||!enabled()||!hit(e.clientX,e.clientY))return;
  e.preventDefault();e.stopImmediatePropagation();surface.setPointerCapture(e.pointerId);
  pointer=e.pointerId;stroke.begin(e.clientX,e.clientY,performance.now());actions.clear();surface.style.cursor=cursor;onStart();
 },true);
 surface.addEventListener('pointermove',e=>{
  const inside=hit(e.clientX,e.clientY);
  if(pointer===e.pointerId){e.preventDefault();e.stopImmediatePropagation();
   if(!enabled()||!(e.buttons&1)){reset();return;}
   if(stroke.move(e.clientX,e.clientY,performance.now(),inside)){lastMotion=performance.now();actions.touch();react(stroke.sample(lastMotion),lastMotion);}
  }else if(!e.buttons)surface.style.cursor=enabled()&&inside?cursor:'grab';
 },true);
 surface.addEventListener('pointerup',e=>{
  if(pointer!==e.pointerId)return;e.preventDefault();e.stopImmediatePropagation();
  const result=enabled()?stroke.finish(performance.now()):null;stroke.reset();pointer=null;
  if(surface.hasPointerCapture(e.pointerId))surface.releasePointerCapture(e.pointerId);
  surface.style.cursor=enabled()&&hit(e.clientX,e.clientY)?cursor:'grab';
  react(result,performance.now());
 },true);
 surface.addEventListener('pointercancel',reset);surface.addEventListener('lostpointercapture',()=>{if(pointer!==null)reset();});
 addEventListener('blur',reset);document.addEventListener('visibilitychange',()=>{if(document.hidden)reset();});
 surface.title='Hold the left mouse button and gently stroke Robby’s head. Alt-drag or drag the base to orbit.';
 return {stroke,hit,reset,
  frame(now,base,reduced=false){const dt=Math.min(100,Math.max(0,now-lastTick));lastTick=now;
   if(!enabled()){reset();return null;}
   const target=now-lastMotion<280?1:0;strength+=(target-strength)*(1-Math.exp(-dt/(target?140:650)));
   if(strength<.005)return null;
   // sampleAction's smooth envelope supplies an adjustable blend without
   // taking over the action queue or preventing speech/stop commands.
   const v=sampleAction({kind:'pet'},.2,base,reduced,actions);
   return {...v,pose:mix(base,v.pose,strength),motion:{...v.motion,tilt:v.motion.tilt*strength},strength};
  }};
}
