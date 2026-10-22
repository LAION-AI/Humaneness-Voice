// Robby desk prototype. Original procedural geometry, Apache-2.0.
// Software 3D projection + Canvas 2D: no WebGL, CDN, mesh download or GPU needed.
// Millimetre proportions follow the supplied 20 Oct 2026 proposal, NOT buildable CAD.
import {clamp,pose,mix} from './emotions.js';
import {geometry,normalizeControls,eyeFill} from './rig.js';
import {SoftwareSurface} from './software-surface.js';

export const DIMENSIONS=Object.freeze({headDiameter:190,headDepth:110,baseDiameter:220,
 baseHeight:70,displayDiameter:150,activeDisplayDiameter:127.6,mountingPitch:75,
 cameraAperture:9,totalHeight:315,tiltPivotHeight:118,panLimit:90,tiltLimit:15,roll:0});
const PI=Math.PI,rad=x=>x*PI/180;
const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]);
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>{const n=Math.hypot(...a)||1;return a.map(v=>v/n);};
const ry=(p,a)=>[p[0]*Math.cos(a)+p[2]*Math.sin(a),p[1],-p[0]*Math.sin(a)+p[2]*Math.cos(a)];
const rx=(p,a)=>[p[0],p[1]*Math.cos(a)-p[2]*Math.sin(a),p[1]*Math.sin(a)+p[2]*Math.cos(a)];
const WHITE=[242,241,237],DARK=[37,43,50],BLACK=[19,25,32],BLUE=[138,211,252];
function lathe(profile,axis='y',group='base',center=[0,0,0],color=WHITE,n=32,part=''){
 if(part==='rear shell'||part==='base shell'||part==='bottom plate'||part==='neck fairing'){
  // Subdivide curved engineering profiles, not the deliberate steps/ports.
  const original=profile;profile=[];
  for(let j=0;j<original.length-1;j++)for(let k=0;k<3;k++){
   const p=original[Math.max(0,j-1)],q=original[j],r=original[j+1],s=original[Math.min(original.length-1,j+2)],t=k/3;
   profile.push(q.map((v,i)=>clamp(.5*((2*v)+(-p[i]+r[i])*t+(2*p[i]-5*v+4*r[i]-s[i])*t*t+(-p[i]+3*v-3*r[i]+s[i])*t*t*t),Math.min(v,r[i]),Math.max(v,r[i]))));
  }profile.push(original.at(-1));
 }
 n=Math.max(n,part.includes('shell')||part.includes('trim')||part==='LCD bezel'||part==='bottom plate'?96:48);
 const points=[],polys=[],normals=[];
 for(let j=0;j<profile.length;j++)for(let i=0;i<n;i++){
  const [h,r]=profile[j],prev=profile[Math.max(0,j-1)],next=profile[Math.min(profile.length-1,j+1)],dh=next[0]-prev[0],dr=next[1]-prev[1];
  const a=2*PI*i/n,c=Math.cos(a)*r,s=Math.sin(a)*r;
  points.push(add(axis==='y'?[c,h,s]:axis==='z'?[c,s,h]:[h,c,s],center));
  normals.push(unit(axis==='y'?[dh*Math.cos(a),-dr,dh*Math.sin(a)]:axis==='z'?[dh*Math.cos(a),dh*Math.sin(a),-dr]:[-dr,dh*Math.cos(a),dh*Math.sin(a)]));
 }
 for(let j=0;j<profile.length-1;j++)for(let i=0;i<n;i++){
  const a=j*n+i,b=(j+1)*n+i,c=(j+1)*n+(i+1)%n,d=j*n+(i+1)%n;
  polys.push(axis==='y'?[a,b,c,d]:[a,d,c,b]);
 }
 return {points,polys,normals,group,color,part};
}
function box(center,size,group='base',color=WHITE,part=''){
 const [x,y,z]=size.map(v=>v/2),points=[[-x,-y,-z],[x,-y,-z],[x,y,-z],[-x,y,-z],[-x,-y,z],[x,-y,z],[x,y,z],[-x,y,z]].map(p=>add(p,center));
 return {points,polys:[[4,5,6,7],[3,2,1,0],[0,4,7,3],[5,1,2,6],[3,7,6,2],[0,1,5,4]],group,color,part};
}
function statusRibbon(){
 const points=[],normals=[],polys=[];
 for(let i=0;i<=32;i++){
  const x=-21+42*i/32,a=x/110,cap=Math.max(0,Math.abs(x)-18.75),h=Math.sqrt(Math.max(0,2.25**2-cap**2));
  for(const y of [31-h,31+h]){points.push([110*Math.sin(a),y,110*Math.cos(a)]);normals.push([Math.sin(a),0,Math.cos(a)]);}
  if(i)polys.push([2*i-2,2*i,2*i+1,2*i-1]);
 }
 return {points,polys,normals,group:'base',color:BLUE,part:'curved capsule status light'};
}
export function buildDeskGeometry(){
 const meshes=[
  // Both materials meet at EXACTLY the same 96-vertex ring (y=12,r=109.5).
  // No intersecting sleeve or mismatched polygons at the black/white seam.
  lathe([[0,0],[0,96],[.8,102],[2,106.5],[4,109],[7,110],[10,110],[12,109.5],[12,0]],'y','base',[0,0,0],DARK,96,'bottom plate'),
  lathe([[12,109.5],[16,110],[27,109.5],[43,107],[57,102],[65,95],[69,85],[70,68],[70,50]],'y','base',[0,0,0],WHITE,96,'base shell'),
  lathe([[69,0],[69,55],[73,55],[76,51],[77,0]],'y','base',[0,0,0],DARK,40,'pan bearing'),
  lathe([[74,49],[76,49],[77,48]],'y','base',[0,0,0],BLUE,40,'pan LED'),
  lathe([[76,0],[76,42],[88,42],[92,34],[92,0]],'y','pan',[0,0,0],BLACK,32,'yaw stage'),
  lathe([[76,0],[76,34],[80,40],[92,42],[103,39],[113,29],[119,18],[119,0]],'y','pan',[0,0,0],WHITE,64,'neck fairing'),
  lathe([[-36,0],[-36,23],[36,23],[36,0]],'x','pan',[0,118,0],DARK,32,'pitch axle'),
  lathe([[-42,0],[-42,25],[-35,25],[-35,0]],'x','pan',[0,118,0],WHITE,32,'pitch cap left'),
  lathe([[35,0],[35,25],[42,25],[42,0]],'x','pan',[0,118,0],WHITE,32,'pitch cap right'),
  lathe([[-44,15.5],[-44,18],[-43,18]],'x','pan',[0,118,0],BLUE,32,'pitch indicator'),
  lathe([[43,18],[44,18],[44,15.5]],'x','pan',[0,118,0],BLUE,32,'pitch indicator'),
  box([-22,-96,0],[10,29,32],'head',WHITE,'tilt adapter'),box([22,-96,0],[10,29,32],'head',WHITE,'tilt adapter'),
  lathe([[-60,0],[-57,25],[-48,55],[-34,78],[-15,91],[10,95],[29,93],[41,88],[47,81],[49,75.5]],'z','head',[0,0,0],WHITE,48,'rear shell'),
  lathe([[44,86],[49,83],[51,75.5],[46,75.5]],'z','trim',[0,0,0],WHITE,48,'front trim'),
  lathe([[48,75],[53,75],[54,63.8],[50,63.8],[48,75]],'z','display',[0,0,0],BLACK,48,'LCD bezel'),
  lathe([[-100,0],[-100,22],[-98,24],[-95,22],[-95,0]],'x','head',[0,-3,-3],DARK,32,'head cap left'),
  lathe([[95,0],[95,22],[98,24],[100,22],[100,0]],'x','head',[0,-3,-3],DARK,32,'head cap right'),
  lathe([[-101,16],[-101,18],[-100.3,18]],'x','head',[0,-3,-3],BLUE,32,'left light ring'),
  lathe([[100.3,18],[101,18],[101,16]],'x','head',[0,-3,-3],BLUE,32,'right light ring'),
  lathe([[32,0],[32,4.5],[36,4.5],[36,0]],'z','head',[0,86,14],BLACK,24,'camera at twelve oclock'),
  lathe([[36.1,0],[36.1,2],[36.5,0]],'z','head',[0,86,14],[42,78,103],20,'camera lens'),
  statusRibbon(),
  box([0,17,109],[5,1.5,1],'base',DARK,'USB microphone opening'),
  box([0,34,-100],[18,9,6],'base',DARK,'rear power port'),
  box([0,0,29],[65,48,8],'mount',[38,91,110],'controller board')
 ];
 meshes.push(box([-12,-18,-59],[15,7,5],'head',DARK,'rear HDMI opening'),
  box([12,-18,-59],[8,5,5],'head',DARK,'rear USB-C opening'));
 for(const x of [-37.5,37.5])for(const y of [-37.5,37.5]){
  meshes.push(lathe([[30,0],[30,5],[47,5],[47,0]],'z','mount',[x,y,0],[180,183,182],16,'M4 standoff 75 x 75'));
 }
 for(let i=-3;i<=3;i++)meshes.push(box([i*8,41,-102],[3,14,2],'base',DARK,'vent'));
 return meshes;
}
function pathCurve(c,points,closed=false){
 if(!points.length)return;c.beginPath();c.moveTo(...points[0]);
 const at=j=>points[closed?(j+points.length)%points.length:Math.max(0,Math.min(points.length-1,j))];
 for(let i=0;i<(closed?points.length:points.length-1);i++){
  const p=at(i-1),q=at(i),r=at(i+1),s=at(i+2);
  c.bezierCurveTo(q[0]+(r[0]-p[0])/6,q[1]+(r[1]-p[1])/6,r[0]-(s[0]-q[0])/6,r[1]-(s[1]-q[1])/6,...r);
 }if(closed)c.closePath();
}
function paintFace(canvas,p,controls,showPoints){
 const c=canvas.getContext('2d'),n=canvas.width;c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,n,n);
 const bg=c.createRadialGradient(n*.35,n*.25,0,n*.5,n*.5,n*.65);bg.addColorStop(0,'#fff');bg.addColorStop(1,'#e9effa');
 c.fillStyle=bg;c.beginPath();c.arc(n/2,n/2,n/2,0,2*PI);c.fill();c.save();c.clip();
 const s=n/450;c.setTransform(s,0,0,s,n/2-400*s,n/2-275*s);
 const g=geometry(p,controls),a=p.actions;
 for(let i=0;i<2;i++){
  pathCurve(c,g.eyes[i],true);c.fillStyle=eyeFill(controls.cartoon);c.fill();
  if(controls.cartoon>.001){c.strokeStyle=`rgba(24,33,41,${controls.cartoon})`;c.lineWidth=3;c.stroke();}
  c.save();c.clip();
  const center=g.pupils[i][0];c.fillStyle='#070c12';c.beginPath();c.ellipse(center[0],center[1],28*a.pupil_scale,33*a.pupil_scale,0,0,2*PI);c.fill();
  c.fillStyle='#fff';c.beginPath();c.arc(center[0]-8,center[1]-15,6,0,2*PI);c.fill();c.restore();
  pathCurve(c,g.brows[i]);c.strokeStyle='#11171d';c.lineWidth=8;c.lineCap='round';c.stroke();
  if(p.blush>.08){c.fillStyle=`rgba(233,155,148,${p.blush*.3})`;c.beginPath();c.ellipse(i?535:265,326,30,15,0,0,2*PI);c.fill();}
 }
 pathCurve(c,g.mouth,true);c.fillStyle='#141b24';c.fill();c.save();c.clip();
 if(p.teeth>.1){c.fillStyle=`rgba(255,255,255,${p.teeth})`;c.fillRect(315,339,170,12);}
 if(p.tongue>.1){c.fillStyle=`rgba(224,135,148,${p.tongue})`;c.beginPath();c.ellipse(400,382,28,11,0,0,2*PI);c.fill();}c.restore();
 if(a.nose_lift>.3){pathCurve(c,g.nose);c.lineWidth=2;c.strokeStyle='#6e7882';c.stroke();}
 if(showPoints){c.fillStyle='#26a5b9';for(const [x,y] of g.points.slice(12,89)){c.beginPath();c.arc(x,y,2,0,2*PI);c.fill();}}
 c.restore();return g;
}
function affineImage(ctx,texture,screen,uv){
 const [[x1,y1],[x2,y2],[x3,y3]]=screen,[[u1,v1],[u2,v2],[u3,v3]]=uv;
 const den=(u1-u3)*(v2-v3)-(u2-u3)*(v1-v3);if(Math.abs(den)<1e-9)return;
 const a=((x1-x3)*(v2-v3)-(x2-x3)*(v1-v3))/den,c=((x2-x3)*(u1-u3)-(x1-x3)*(u2-u3))/den;
 const b=((y1-y3)*(v2-v3)-(y2-y3)*(v1-v3))/den,d=((y2-y3)*(u1-u3)-(y1-y3)*(u2-u3))/den;
 // Slightly overlap the clipping edges. Antialiased adjacent triangle clips
 // otherwise leave dark hairline cracks across the LCD on software canvases.
 const center=[(x1+x2+x3)/3,(y1+y2+y3)/3];
 const expanded=screen.map(p=>{const dx=p[0]-center[0],dy=p[1]-center[1],n=Math.hypot(dx,dy)||1;return [p[0]+dx/n,p[1]+dy/n];});
 ctx.save();ctx.beginPath();ctx.moveTo(...expanded[0]);ctx.lineTo(...expanded[1]);ctx.lineTo(...expanded[2]);ctx.closePath();ctx.clip();
 ctx.transform(a,b,c,d,x3-a*u3-c*v3,y3-b*u3-d*v3);ctx.drawImage(texture,0,0);ctx.restore();
}

export function createDeskRobotRig(container){
 const canvas=document.createElement('canvas');canvas.className='desk-robot-canvas';canvas.setAttribute('aria-label','Three-dimensional two-axis desk robot; drag to orbit');
 canvas.setAttribute('role','img');container.replaceChildren(canvas);
 const ctx=canvas.getContext('2d',{alpha:true}),faceCanvas=document.createElement('canvas');faceCanvas.width=faceCanvas.height=512;
 const faceCtx=faceCanvas.getContext('2d',{willReadFrequently:true}),surface=new SoftwareSurface();
 if(!ctx)throw Error('This browser needs Canvas 2D support for the desk robot.');
 // The display centre is at y=220. Place the neutral viewer on that eye
 // line, not at the chassis centre or an exhibition-style 12-degree orbit.
 const meshes=buildDeskGeometry(),state={pan:0,tilt:0,roll:0,view:0,elevation:0,distance:650,
  manual:false,manualPan:0,manualTilt:0,exploded:0,blueprint:false,tour:false,fps:30};
 for(const mesh of meshes)mesh.faces=mesh.polys.flatMap(indices=>{
  const p=indices.map(i=>mesh.points[i]);let n=[0,0,0];
  for(let i=1;i<p.length-1;i++){n=cross(sub(p[i],p[0]),sub(p[i+1],p[0]));if(Math.hypot(...n)>1e-5)break;}
  return Math.hypot(...n)>1e-5?[{indices,normal:unit(n)}]:[];
 });
 let cp=1,sp=0,ct=1,st=0,cv=1,sv=0,ce=1,se=0,bodyCache=null,actionPan=0,actionTilt=0;
 let smooth=pose(),controls=normalizeControls(),last={},showPoints=false,lastRender=-1,renderCount=0,totalRenderMs=0,durations=[],disposed=false;
 let width=500,height=490,drag=null;
 const ro=new ResizeObserver(()=>{const rect=container.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);
  const ratio=Math.min(devicePixelRatio||1,1.5),w=Math.round(width*ratio),h=Math.round(height*ratio);
  // Assigning even the SAME canvas width clears the backing store. Never
  // blank a frame on redundant ResizeObserver notifications.
  if(canvas.width!==w)canvas.width=w;if(canvas.height!==h)canvas.height=h;
  canvas.style.width=width+'px';canvas.style.height=height+'px';surface.resize(w,h);});ro.observe(container);
 function rotateNormal(p,group){let q=p;
  if(['head','trim','display','mount'].includes(group))q=[p[0],p[1]*ct+p[2]*st,-p[1]*st+p[2]*ct];
  return group==='base'?q:[q[0]*cp+q[2]*sp,q[1],-q[0]*sp+q[2]*cp];
 }
 function transform(p,group){
  let q=p;
  if(['head','trim','display','mount'].includes(group)){
   q=[p[0],p[1],p[2]+(group==='trim'?state.exploded*40:group==='display'?state.exploded*23:group==='mount'?state.exploded*10:-state.exploded*18)];
   const y=q[1]+102,z=q[2];q=[q[0],y*ct+z*st+118,-y*st+z*ct];
  }
  if(group!=='base')q=[q[0]*cp+q[2]*sp,q[1],-q[0]*sp+q[2]*cp];
  if(group==='base'&&state.exploded)q=[q[0],q[1]-state.exploded*(q[1]<13?22:0),q[2]];
  return q;
 }
 function viewVector(p){const x=p[0]*cv-p[2]*sv,z=p[0]*sv+p[2]*cv;return [x,p[1]*ce-z*se,p[1]*se+z*ce];}
 function view(p){return viewVector([p[0],p[1]-220,p[2]]);}
 function project(p){const q=view(p),depth=Math.max(100,state.distance-q[2]),s=Math.min(width/360,height/380)*650/depth;
  return [width*.5+q[0]*s,height*.28-q[1]*s,depth];}
 function facing(world,normal){const p=view(world),n=viewVector(normal);return dot(n,[-p[0],-p[1],state.distance-p[2]])>0;}
 const light=unit([-.5,.8,.75]),halfLight=unit([-.25,.4,1]);
 function render(time){
  const pan=rad(Math.round((state.pan+actionPan)*50)/50),tilt=rad(Math.round(clamp(state.tilt+actionTilt,-15,15)*50)/50);
  cp=Math.cos(pan);sp=Math.sin(pan);ct=Math.cos(tilt);st=Math.sin(tilt);
  cv=Math.cos(state.view);sv=Math.sin(state.view);ce=Math.cos(state.elevation);se=Math.sin(state.elevation);
  const started=performance.now(),ratio=canvas.width/width;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  const foot=project([0,0,0]);ctx.save();ctx.translate(foot[0],foot[1]);ctx.scale(1,.18);
  const shadow=ctx.createRadialGradient(0,0,5,0,0,width*.29);shadow.addColorStop(0,'#41516b30');shadow.addColorStop(1,'#41516b00');ctx.fillStyle=shadow;ctx.beginPath();ctx.arc(0,0,width*.29,0,2*PI);ctx.fill();ctx.restore();
  if(state.blueprint){ctx.strokeStyle='#93a8b72b';ctx.lineWidth=.7;for(let i=-150;i<=150;i+=50){for(const line of [[[i,0,-150],[i,0,150]],[[-150,0,i],[150,0,i]]]){ctx.beginPath();ctx.moveTo(...project(line[0]).slice(0,2));ctx.lineTo(...project(line[1]).slice(0,2));ctx.stroke();}}}
  surface.resize(canvas.width,canvas.height);surface.begin();let visibleFaces=0;
  const cacheKey=[pan,tilt,state.view,state.elevation,state.distance,state.exploded,canvas.width,canvas.height].join('/');
  const lighting=n=>{const diffuse=Math.max(0,dot(n,light)),shine=Math.pow(Math.max(0,dot(n,halfLight)),20);return .62+.32*diffuse+.06*shine;};
  if(bodyCache?.key===cacheKey){surface.bytes.set(bodyCache.pixels);surface.depth.set(bodyCache.depth);visibleFaces=bodyCache.visibleFaces;}
  else {for(const mesh of meshes){
   const world=mesh.points.map(p=>transform(p,mesh.group)),screen=world.map(p=>{const s=project(p);return [s[0]*ratio,s[1]*ratio,s[2]];});
   const colors=mesh.normals?.map(n=>{const lum=lighting(rotateNormal(n,mesh.group));return mesh.color.map(v=>v*lum);});
   for(const face of mesh.faces){
    const {indices}=face,normal=rotateNormal(face.normal,mesh.group);
    if(!facing(world[indices[0]],normal))continue;
    const flat=mesh.color.map(v=>v*lighting(normal));visibleFaces++;
    for(let j=1;j<indices.length-1;j++){
     const ids=[indices[0],indices[j],indices[j+1]];
     surface.triangle(...ids.map(i=>screen[i]),ids.map(i=>colors?colors[i]:flat));
    }
   }
  }
   bodyCache={key:cacheKey,pixels:surface.bytes.slice(),depth:surface.depth.slice(),visibleFaces};
  }
  const front=transform([0,0,54],'display'),normal=sub(transform([0,0,55],'display'),front);
  if(facing(front,normal)){
   const uv=[[0,0],[512,0],[512,512],[0,512]],texture=faceCtx.getImageData(0,0,512,512);
   const s=uv.map(([u,v])=>{const p=project(transform([-63.8+u/512*127.6,63.8-v/512*127.6,54.2],'display'));return [p[0]*ratio,p[1]*ratio,p[2]];});
   for(const ids of [[0,1,2],[0,2,3]])surface.triangle(...ids.map(i=>s[i]),null,texture,ids.map(i=>uv[i]));
  }
  ctx.drawImage(surface.finish(),0,0,width,height);
  if(showPoints||state.blueprint){
   for(const [a,b,color,label] of [[[0,70,0],[0,165,0],'#328ba7','PAN'],[[-67,118,0],[67,118,0],'#be8150','TILT']]){
    ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(...project(transform(a,'pan')).slice(0,2));ctx.lineTo(...project(transform(b,'pan')).slice(0,2));ctx.stroke();ctx.setLineDash([]);
    const p=project(transform(b,'pan'));ctx.fillStyle=color;ctx.font='10px system-ui';ctx.fillText(label,p[0]+4,p[1]-4);
   }
  }
  const ms=performance.now()-started;renderCount++;totalRenderMs+=ms;durations.push(ms);if(durations.length>120)durations.shift();
  const sorted=[...durations].sort((a,b)=>a-b);
   last={...last,pan:state.pan,tilt:state.tilt,visualPan:state.pan+actionPan,visualTilt:clamp(state.tilt+actionTilt,-15,15),roll:0,dimensions:DIMENSIONS,renderer:'CPU z-buffer · smooth vertex normals · no WebGL',
   meshFaces:meshes.reduce((n,m)=>n+m.polys.length,0),visibleFaces,renderCount,triangles:surface.triangles,
   smoothShading:true,depthBuffer:true,controls:{...controls},
   renderMs:ms,renderP50Ms:sorted[Math.floor(sorted.length*.5)],renderP95Ms:sorted[Math.floor(sorted.length*.95)],
   meanRenderMs:totalRenderMs/renderCount,viewDegrees:state.view*180/PI,elevationDegrees:state.elevation*180/PI,
   eyeContactErrorDegrees:Math.acos(clamp(dot(unit(normal),unit(sub(add(ry(rx([0,0,state.distance],-state.elevation),state.view),[0,220,0]),front))),-1,1))*180/PI,
   exploded:state.exploded,expression:smooth.slug};
 }
 canvas.style.touchAction='none';canvas.style.cursor='grab';
 canvas.onpointerdown=e=>{if(state.gazeCalibration||e.defaultPrevented)return;drag=[e.clientX,e.clientY,state.view,state.elevation];canvas.setPointerCapture(e.pointerId);canvas.style.cursor='grabbing';};
 canvas.onpointermove=e=>{if(drag){state.view=drag[2]+(e.clientX-drag[0])*.008;state.elevation=clamp(drag[3]+(e.clientY-drag[1])*.005,rad(-10),rad(40));}};
 canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;canvas.style.cursor='grab';};
 canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=clamp(state.distance+e.deltaY*.4,450,950);},{passive:false});
 canvas.tabIndex=0;canvas.onkeydown=e=>{if(state.gazeCalibration)return;if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home'].includes(e.key)){e.preventDefault();if(e.key==='Home'){state.view=0;state.elevation=0;}else if(e.key==='ArrowLeft')state.view-=rad(10);else if(e.key==='ArrowRight')state.view+=rad(10);else state.elevation=clamp(state.elevation+rad(e.key==='ArrowUp'?5:-5),rad(-10),rad(40));}};
 return {state,idlePose(){const p=pose();p.actions.brow_arch=.42;p.actions.smile=.62;return p;},showKeypoints(on){showPoints=!!on;},
  headRegion(){const center=project(transform([0,0,0],'head'));
   const scale=Math.min(width/360,height/380)*650/center[2];
   return {x:center[0],y:center[1],rx:98*scale,ry:98*scale};},
  setManual(pan,tilt,on=true){state.manual=on;state.manualPan=clamp(+pan,-90,90);state.manualTilt=clamp(+tilt,-15,15);},
  setExploded(value){state.exploded=clamp(+value,0,1);},setBlueprint(on){state.blueprint=!!on;},setTour(on){state.tour=!!on;},
  center(){state.view=0;state.elevation=0;state.distance=650;state.manual=false;state.tour=false;},
  update(input={},dtMs=16.7,timeMs=0){if(disposed)return last;
   const motion=input.actionMotion||{};actionPan=clamp(motion.pan,-20,20)+clamp(motion.spin,-1080,1080);actionTilt=clamp(motion.tilt,-12,12);
   const dt=clamp(dtMs,0,100),alpha=1-Math.exp(-dt/160),target=input.pose||pose();smooth=input.freezeExpression?target:{...mix(smooth,target,alpha),slug:target.slug};
   const c=normalizeControls(input);for(const key in controls)controls[key]+=alpha*(c[key]-controls[key]);
   const gesture=smooth.gesture;
   const gaze=input.gazeTarget,eyeContact=gaze&&Number.isFinite(gaze.pan)&&Number.isFinite(gaze.tilt);
   const panTarget=state.manual?state.manualPan:state.tour?Math.sin(timeMs/1800)*45:eyeContact?clamp(gaze.pan,-65,65):clamp(controls.x*52+gesture[0]*.10+Math.sin(timeMs/1700)*gesture[6]*7,-70,70);
   const tiltTarget=state.manual?state.manualTilt:state.tour?Math.sin(timeMs/2700)*12:eyeContact?clamp(gaze.tilt,-15,15):clamp(-controls.y*15-gesture[1]*.55,-15,15);
   state.pan=clamp(state.pan+clamp((panTarget-state.pan)*alpha,-45*dt/1000,45*dt/1000),-90,90);
   state.tilt=clamp(state.tilt+clamp((tiltTarget-state.tilt)*alpha,-25*dt/1000,25*dt/1000),-15,15);
   const clock=input.freezeExpression?performance.now():timeMs||performance.now();
   const interval=1000/state.fps,elapsed=clock-lastRender;
   if(dtMs<100&&elapsed<interval)return last;
   // Carry frame-time remainder: discarding it often turns a 30-fps budget
   // into 20 fps on a slightly jittery 60-Hz software/compositor clock.
   lastRender=dtMs>=100?clock:clock-elapsed%interval;
   // While maintaining eye contact, emotion acts through lids/brows/mouth,
   // not an unrelated sideways gaze. The pupils make a SMALL lead movement
   // while the slower mechanical head catches up, then return to centre.
   const displayPose=eyeContact&&!input.actionEyes?{...smooth,actions:{...smooth.actions,gaze_x:clamp((panTarget-state.pan)/35,-.35,.35),gaze_y:clamp(-(tiltTarget-state.tilt)/25,-.3,.3)}}:smooth;
   const displayControls=eyeContact?{...controls,x:0,y:0}:controls;
   const g=paintFace(faceCanvas,displayPose,displayControls,showPoints);last={...last,pose:smooth,displayPose,geometry:g,gazeActive:!!eyeContact};render(timeMs);return last;
  },snapshot(){return last;},reset(){smooth=pose();controls=normalizeControls();this.center();this.update({},1000,0);},
  dispose(){disposed=true;ro.disconnect();canvas.remove();}
 };
}
