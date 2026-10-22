import {profiles} from './emotions.js';
import {GRIMACES} from './avatar-actions.js';
export function attachActionControls(actions){
 const panel=document.createElement('details');panel.className='desk-controls';panel.id='avatar-action-controls';
 panel.innerHTML=`<summary>Personality &amp; little tricks</summary>
 <p>Ask “Nod”, “Roll your eyes”, “Turn 360 degrees counterclockwise”, or “Look angry, then make a funny face”. These are screen animations, not motor commands.</p>
 <label><input id="avatar-actions-enabled" type="checkbox" checked> Allow expressive actions</label>
 <label><input id="avatar-idle-enabled" type="checkbox" checked> Play after a minute of quiet · silly faces and a two-turn carousel</label>
 <label><input id="avatar-petting-enabled" type="checkbox" checked> Stroke Robby’s head with the left mouse button</label>
 <label><input id="avatar-pet-speech-enabled" type="checkbox" checked> Occasional LLM-generated reactions to strokes</label>
 <label><input id="avatar-pet-invites-enabled" type="checkbox" checked> Rare, optional invitations to stroke his head</label>
 <label><input id="avatar-listening-enabled" type="checkbox" checked> Quiet listening nods / head shakes chosen by the decision model</label>
 <label for="avatar-trick">Try a silent action</label><select id="avatar-trick"></select>
 <div class="desk-buttons"><button type="button" id="avatar-perform">Perform</button><button type="button" id="avatar-stop">Stop animation</button></div>
 <p id="avatar-action-status" aria-live="polite">Camera tracking resumes automatically after each trick.</p>
 <p>Hold left-click and move gently across the head; a hand cursor shows where. Alt-drag or drag the base to orbit. Clicks or holding still do not count as strokes. Robby is a playful AI character, not a creature with physical touch sensations. Spoken reactions need a connected conversation. Invitations never repeat in an unanswered quiet spell; reduced motion disables carousel play.</p>
 <p>Listening cues reuse enabled live usefulness/rule decisions and require a recognized speaker.</p>
 <a href="/static/avatar-manual.html" target="_blank" rel="noopener">Illustrated manual · all 40 expressions ↗</a> · <a href="https://laion-ai.github.io/Humaneness-Voice/robby/" target="_blank" rel="noopener">GitHub Pages copy ↗</a>`;
 document.getElementById('avatar').after(panel);
 const items=[['Nod yes',{kind:'nod'}],['Shake head no',{kind:'shake'}],['360° clockwise (from above)',{kind:'spin',direction:'clockwise',duration_s:4}],['360° counterclockwise (from above)',{kind:'spin',direction:'counterclockwise',duration_s:4}],['Roll eyes',{kind:'eye_roll'}],['Wink',{kind:'wink'}],['Curious peek',{kind:'curious'}],
  ...GRIMACES.slice(0,4).map(variant=>['Cute grimace: '+variant,{kind:'grimace',variant}]),
  ...profiles.map(p=>[p.label,{kind:'emotion',emotion:p.slug,duration_s:3}]),
  ['Angry → cheeky → content',[{kind:'emotion',emotion:'anger'},{kind:'grimace',variant:'cheeky'},{kind:'emotion',emotion:'contentment'}]],
  ...GRIMACES.slice(4).map(variant=>['Cartoon grimace: '+variant,{kind:'grimace',variant}]),
  ['Carousel · two playful turns',{kind:'carousel',direction:'clockwise',duration_s:4.5}],
  ['Contented kitten-like smile',{kind:'pet',duration_s:3}]];
 const select=panel.querySelector('select');items.forEach(([label],i)=>select.add(new Option(label,i)));
 for(const [id,key] of [['avatar-actions-enabled','enabled'],['avatar-idle-enabled','idleEnabled'],['avatar-listening-enabled','listeningEnabled'],['avatar-petting-enabled','pettingEnabled'],['avatar-pet-speech-enabled','petSpeechEnabled'],['avatar-pet-invites-enabled','petInvitesEnabled']]){
  const box=document.getElementById(id);try{box.checked=localStorage.getItem('robby.'+id)!=='false';}catch{}actions[key]=box.checked;
  box.onchange=()=>{actions[key]=box.checked;actions.clear();try{localStorage.setItem('robby.'+id,String(box.checked));}catch{}};
 }
 document.getElementById('avatar-perform').onclick=()=>{const value=items[+select.value][1];actions.play((Array.isArray(value)?value:[value]).map(s=>({duration_s:2,...s})));};
 document.getElementById('avatar-stop').onclick=()=>actions.clear();
 let last='';const timer=setInterval(()=>{const text=actions.active?`Performing ${actions.active.kind.replaceAll('_',' ')} · ${actions.active.source}`:!actions.enabled?'Expressive actions disabled.':'Camera tracking resumes automatically after each trick.';if(text!==last){document.getElementById('avatar-action-status').textContent=text;last=text;}},250);
 addEventListener('pagehide',()=>clearInterval(timer),{once:true});
 return panel;
}
