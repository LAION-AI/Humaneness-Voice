// MIT. Independent robot-specific 101-anchor rig, not the French-face topology.
import {clamp,pose,mix} from './emotions.js';
export const REQUIRED_IDS=['body','head','arm-left','arm-right','brow-left','brow-right','mouth-shape',
 'eye-outline-left','eye-outline-right','eye-mask-left','eye-mask-right','iris-left','iris-right',
 'mouth-mask','teeth','tongue','nose','cheek-left','cheek-right','blush-left','blush-right','keypoints','body-keypoints'];
export function normalizeControls(c={}){return {x:clamp(c.x,-1,1),y:clamp(c.y,-1,1),roll:clamp(c.roll,-35,35),
 blinkLeft:clamp(c.blinkLeft,0,1),blinkRight:clamp(c.blinkRight,0,1),smile:clamp(c.smile,0,1),jaw:clamp(c.jaw,0,1),audioLevel:clamp(c.audioLevel,0,1)};}
export function controlsFromFace(result){const p=result?.faceLandmarks?.[0];if(!p?.[263])return normalizeControls();
 const b=Object.fromEntries((result.faceBlendshapes?.[0]?.categories||[]).map(x=>[x.categoryName,x.score]));
 return normalizeControls({x:(.5-p[1].x)*4,y:(p[1].y-.5)*3,roll:Math.atan2(p[263].y-p[33].y,p[263].x-p[33].x)*180/Math.PI,
 blinkLeft:b.eyeBlinkRight,blinkRight:b.eyeBlinkLeft,smile:((b.mouthSmileLeft||0)+(b.mouthSmileRight||0))/2,jaw:b.jawOpen});}
function curve(points,closed=false){if(!points.length)return '';let d=`M${points[0][0]} ${points[0][1]}`;
 for(let i=0;i<(closed?points.length:points.length-1);i++){const at=j=>points[closed?(j+points.length)%points.length:Math.max(0,Math.min(points.length-1,j))];
 const p=at(i-1),q=at(i),r=at(i+1),s=at(i+2);
 d+=`C${q[0]+(r[0]-p[0])/6} ${q[1]+(r[1]-p[1])/6} ${r[0]-(s[0]-q[0])/6} ${r[1]-(s[1]-q[1])/6} ${r[0]} ${r[1]}`;}
 return d+(closed?'Z':'');}
export const KEYPOINT_IDS=[...Array.from({length:12},(_,i)=>'shell_'+i),
 ...['left','right'].flatMap(s=>Array.from({length:12},(_,i)=>'eye_'+s+'_'+i)),
 ...['left','right'].flatMap(s=>Array.from({length:5},(_,i)=>'pupil_'+s+'_'+i)),
 ...['left','right'].flatMap(s=>Array.from({length:5},(_,i)=>'brow_'+s+'_'+i)),
 ...Array.from({length:20},(_,i)=>'mouth_'+i),...Array.from({length:5},(_,i)=>'nose_'+i),
 ...['left','right'].flatMap(s=>Array.from({length:4},(_,i)=>'cheek_'+s+'_'+i)),...Array.from({length:12},(_,i)=>'body_'+i)];
export function geometry(p=pose(),controls={}){
 const a=p.actions,c=normalizeControls(controls),eyes=[],brows=[],pupils=[],cheeks=[];
 for(let side=0;side<2;side++){
  const sign=side?-1:1,cx=side?530:270,blink=side?c.blinkRight:c.blinkLeft;
  const openness=clamp(a.eye_open+sign*a.eye_asym*.65,.025,1.6)*(1-blink*.97);
  const h=clamp(43*openness-12*a.squint,1.2,68),rx=37,tilt=a.lid_tilt*sign*.28;
  eyes.push(Array.from({length:12},(_,j)=>{const t=2*Math.PI*j/12,x=rx*Math.cos(t),y=h*Math.sin(t);
   return [cx+x,267+y+tilt*x-(y<0?a.lid_curve*7*Math.sin(t)**2:0)];}));
  const gx=clamp(a.gaze_x+c.x*.25,-1,1)*13,gy=clamp(a.gaze_y+c.y*.2,-1,1)*13,radius=clamp(a.pupil_scale,.5,1.5)*28;
  pupils.push([[cx+gx,267+gy],[cx+gx-radius,267+gy],[cx+gx,267+gy-radius*1.18],[cx+gx+radius,267+gy],[cx+gx,267+gy+radius*1.18]]);
  brows.push(Array.from({length:5},(_,j)=>{const u=j/4,x=(u-.5)*80,inner=side?1-u:u;
   return [cx+x,190-a.brow_raise*25-a.brow_asym*sign*18-a.brow_inner*24*inner-a.brow_outer*19*(1-inner)
    -a.brow_arch*15*Math.sin(u*Math.PI)+a.frown*20*inner];}));
  const y=327-a.cheek_lift*12;cheeks.push([[cx-37,y+9],[cx-12,y],[cx+13,y-1],[cx+36,y+9]]);
 }
 const width=clamp(a.mouth_width,.35,1.7)*51,open=clamp(a.mouth_open,0,1)*37+Math.max(c.audioLevel*26,c.jaw*20),mouth=[],my=354+a.mouth_y*15;
 for(let side=0;side<2;side++)for(let j=0;j<10;j++){
  const u=side?1-j/9:j/9,x=(u*2-1)*width,center=my+a.smile*(15*(1-(u*2-1)**2)-10)+a.mouth_asym*(u-.5)*25;
  mouth.push([400+x,center+Math.sin(u*Math.PI)*Math.max(1.2,open)*(side?1:-.55)+a.mouth_tension*Math.sin(u*2*Math.PI)*3]);
 }
 const ny=316-a.nose_lift*9,nose=[[388,ny+2],[394,ny-2],[400,ny],[406,ny-2],[412,ny+2]];
 const shell=Array.from({length:12},(_,i)=>[400+225*Math.cos(i*Math.PI/6),265+205*Math.sin(i*Math.PI/6)]);
 const body=[[400,482],[180,540],[620,540],[180,660],[620,660],[203,743],[597,743],[270,805],[530,805],[375,835],[425,835],[400,875]];
 const points=[...shell,...eyes.flat(),...pupils.flat(),...brows.flat(),...mouth,...nose,...cheeks.flat(),...body];
 return {points,eyes,brows,pupils,mouth,nose,cheeks,open};
}
export function createRobbyRig(svg){const n=Object.fromEntries(REQUIRED_IDS.map(id=>[id,svg.querySelector('#'+id)]));
 const missing=REQUIRED_IDS.filter(id=>!n[id]);if(missing.length)throw new Error('Missing robot anchors: '+missing.join(', '));
 let smooth=pose(),controls=normalizeControls(),visiblePoints=false,last={};
 const dots=KEYPOINT_IDS.map((id,i)=>{const dot=document.createElementNS('http://www.w3.org/2000/svg','circle');dot.setAttribute('r','2.2');dot.setAttribute('data-anchor',id);(i<89?n.keypoints:n['body-keypoints']).append(dot);return dot;});
 return {showKeypoints(on){visiblePoints=on;for(const id of ['keypoints','body-keypoints'])n[id].setAttribute('opacity',on?'1':'0');},
  update(input={},dtMs=16.7,timeMs=0){const alpha=1-Math.exp(-clamp(dtMs,0,1000)/145),target=input.pose||pose();
   smooth={...mix(smooth,target,alpha),slug:target.slug};const c=normalizeControls(input);for(const k in controls)controls[k]+=alpha*(c[k]-controls[k]);
   const g=geometry(smooth,controls),a=smooth.actions;
   for(let i=0;i<2;i++){const side=i?'right':'left',eye=curve(g.eyes[i],true),p=g.pupils[i][0];
    n['eye-outline-'+side].setAttribute('d',eye);n['eye-mask-'+side].setAttribute('d',eye);
    n['iris-'+side].setAttribute('transform',`translate(${p[0]} ${p[1]}) scale(${clamp(a.pupil_scale,.5,1.4)})`);
    n['brow-'+side].setAttribute('d',curve(g.brows[i]));n['cheek-'+side].setAttribute('d',curve(g.cheeks[i]));
    n['cheek-'+side].setAttribute('opacity',String(clamp(a.cheek_lift*.5+a.nose_lift*.2,0,.6)));
    n['blush-'+side].setAttribute('opacity',String(.13+smooth.blush*.65));
   }
   const mouth=curve(g.mouth,true);n['mouth-shape'].setAttribute('d',mouth);n['mouth-mask'].setAttribute('d',mouth);
   n.teeth.setAttribute('opacity',String(smooth.teeth));n.tongue.setAttribute('opacity',String(smooth.tongue));
   n.nose.setAttribute('d',curve(g.nose));n.nose.setAttribute('opacity',String(clamp(a.nose_lift*.9,0,.75)));
   const v=smooth.gesture,beat=Math.sin(timeMs/600*v[7]),breath=Math.sin(timeMs/2100);
   n.head.setAttribute('transform',`translate(${controls.x*7+v[2]} ${breath+v[1]+controls.y*3}) rotate(${v[0]+controls.roll*.3+v[6]*beat*4} 400 440)`);
   n.body.setAttribute('transform',`translate(${v[2]*.4} ${breath*.6}) rotate(${v[2]*.3} 400 800)`);
   n['arm-left'].setAttribute('transform',`translate(0 ${-v[5]}) rotate(${v[3]+Math.abs(v[3])*.09*beat} 180 550)`);
   n['arm-right'].setAttribute('transform',`translate(0 ${-v[5]}) rotate(${-v[4]-Math.abs(v[4])*.09*beat} 620 550)`);
   if(visiblePoints)g.points.forEach((p,i)=>{
    let [x,y]=p;const part=i-89;
    if(part>=1&&part<=10){const left=part%2===1,pivot=left?180:620,
      degrees=left?v[3]+Math.abs(v[3])*.09*beat:-v[4]-Math.abs(v[4])*.09*beat,
      r=degrees*Math.PI/180,dx=x-pivot,dy=y-550;x=pivot+dx*Math.cos(r)-dy*Math.sin(r);y=550+dx*Math.sin(r)+dy*Math.cos(r)-v[5];}
    dots[i].setAttribute('cx',x);dots[i].setAttribute('cy',y);
   });
   last={pose:smooth,geometry:g};return last;
  },snapshot(){return last;},reset(){smooth=pose();controls=normalizeControls();this.update({},1000,0);}
 };
}
