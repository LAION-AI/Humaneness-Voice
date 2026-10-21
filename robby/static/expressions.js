import {profiles,pose,mix} from './emotions.js';
import {createRobbyRig,geometry,KEYPOINT_IDS} from './rig.js';
const $=id=>document.getElementById(id),source=await fetch('/static/robby.svg').then(r=>r.text());
$('lab-avatar').innerHTML=source;const rig=createRobbyRig($('lab-avatar').querySelector('svg'));let seed=42;
for(const id of ['a','b'])for(const p of [{slug:'neutral',label:'Neutral'},...profiles]){const o=document.createElement('option');o.value=p.slug;o.textContent=p.label;$(id).append(o);}
$('a').value='amusement';$('b').value='teasing';$('seed').onclick=()=>seed++;
$('points').onchange=()=>rig.showKeypoints($('points').checked);
for(const profile of profiles){
 const holder=document.createElement('div');holder.innerHTML=source;const svg=holder.querySelector('svg'),r=createRobbyRig(svg);
 for(let i=0;i<3;i++)r.update({pose:pose(profile.slug,1,0)},1000,0);
 const card=document.createElement('button');card.className='emotion-card';card.dataset.emotion=profile.slug;
 const img=document.createElement('img');img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(new XMLSerializer().serializeToString(svg));img.alt=profile.label;
 const label=document.createElement('span');label.textContent=profile.id+'. '+profile.label;
 const family=document.createElement('small');family.textContent=profile.family;
 card.append(img,label,family);card.onclick=()=>{$('a').value=profile.slug;$('blend').value=0;window.scrollTo({top:80,behavior:'smooth'});};$('cards').append(card);
}
let last=0;
function frame(t){const blend=Number($('blend').value),strength=Number($('intensity').value),variation=Number($('variation').value);
 const target=mix(pose($('a').value,strength,variation,seed),pose($('b').value,strength,variation,seed+5),blend);
 rig.update({pose:target},last?t-last:16.7,matchMedia('(prefers-reduced-motion: reduce)').matches?0:t);last=t;
 $('blend-value').textContent=Math.round(blend*100)+'%';$('label').textContent=`${Math.round((1-blend)*100)}% ${$('a').value.replaceAll('_',' ')} · ${Math.round(blend*100)}% ${$('b').value.replaceAll('_',' ')} · seed ${seed}`;requestAnimationFrame(frame);}
requestAnimationFrame(frame);window.robotGallery={rig,geometry,KEYPOINT_IDS,profiles,pose,mix};
