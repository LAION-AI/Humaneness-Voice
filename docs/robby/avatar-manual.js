import {createDeskRobotRig} from './desk-robot3d.js';
import {createRobbyRig} from './rig.js';
import {pose,profiles} from './emotions.js';
import {AvatarActions,GRIMACES} from './avatar-actions.js';
import {attachPetting} from './avatar-petting.js';
const rig=createDeskRobotRig(document.getElementById('manual-avatar')),actions=new AvatarActions();actions.idleEnabled=false;rig.state.fps=20;
const petting=attachPetting({element:document.getElementById('manual-avatar'),rig,actions});
const examples=[['Nod',[{kind:'nod'}]],['Shake head',[{kind:'shake'}]],['360° clockwise',[{kind:'spin',direction:'clockwise',duration_s:4}]],['360° counterclockwise',[{kind:'spin',direction:'counterclockwise',duration_s:4}]],['Roll eyes',[{kind:'eye_roll'}]],['Wink',[{kind:'wink'}]],['Curious peek',[{kind:'curious'}]],...['crossed','opposite','wiggle','cheeky'].map(variant=>['Grimace: '+variant,[{kind:'grimace',variant}]]),['Angry → cheeky → content',[{kind:'emotion',emotion:'anger'},{kind:'grimace',variant:'cheeky'},{kind:'emotion',emotion:'contentment'}]],...profiles.map(p=>[p.label,[{kind:'emotion',emotion:p.slug,duration_s:3}]])];
examples.push(...GRIMACES.slice(4).map(variant=>['Grimace: '+variant,[{kind:'grimace',variant}]]),
 ['Carousel: two turns',[{kind:'carousel',direction:'clockwise',duration_s:4.5}]],['Contented petting pose',[{kind:'pet',duration_s:3}]]);
const select=document.getElementById('manual-action');examples.forEach(([label],i)=>select.add(new Option(label,i)));
const play=steps=>actions.play(steps.map(s=>({duration_s:2,...s})));
document.getElementById('manual-play').onclick=()=>play(examples[+select.value][1]);document.getElementById('manual-stop').onclick=()=>actions.clear();
const markup=await fetch(new URL('./robby.svg',import.meta.url)).then(r=>r.text());
for(const p of profiles){
 const card=document.createElement('article');card.className='card';const figure=document.createElement('div');figure.innerHTML=markup;
 const svg=figure.querySelector('svg');svg.setAttribute('viewBox','145 45 510 455');svg.setAttribute('aria-label',p.label);
 // Scope every SVG ID per thumbnail so clipping never resolves to another card.
 const nodes=[...svg.querySelectorAll('[id]')],ids=nodes.map(n=>n.id);const thumb=createRobbyRig(svg);thumb.update({pose:pose(p.slug,1,0)},1000,0);thumb.update({pose:pose(p.slug,1,0)},1000,0);
 for(const node of svg.querySelectorAll('*'))for(const attr of [...node.attributes]){let value=attr.value;for(const id of ids)value=value.replaceAll('url(#'+id+')','url(#'+p.slug+'-'+id+')');if(attr.name==='id')value=p.slug+'-'+value;node.setAttribute(attr.name,value);}
 const button=document.createElement('button');button.textContent=p.label;button.onclick=()=>{play([{kind:'emotion',emotion:p.slug,duration_s:3}]);document.getElementById('manual-avatar').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'center'});};card.append(figure,button);document.getElementById('expression-cards').append(card);
}
let previous=performance.now(),lastLabel='';function frame(now){const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches,normal=actions.frame(now,rig.idlePose(),{reduced,hidden:document.hidden}),a=petting.frame(now,normal.pose,reduced)||normal;rig.update({...a.controls,pose:a.pose,actionMotion:a.motion,actionEyes:a.eyes},now-previous,now);previous=now;
 const label=actions.active?'Preview: '+actions.active.kind.replaceAll('_',' '):'Ready · drag to orbit · all previews run locally on CPU';if(label!==lastLabel){document.getElementById('manual-status').textContent=label;lastLabel=label;}requestAnimationFrame(frame);}requestAnimationFrame(frame);
window.manualPreview={rig,actions,profiles};
