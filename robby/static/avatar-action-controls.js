import {profiles} from './emotions.js';
import {GRIMACES} from './avatar-actions.js';
export function attachActionControls(actions){
 const panel=document.createElement('details');panel.className='desk-controls';panel.id='avatar-action-controls';
 panel.innerHTML=`<summary>Personality &amp; little tricks</summary>
 <p>Ask “Nod”, “Roll your eyes”, “Turn 360 degrees counterclockwise”, or “Look angry, then make a funny face”. These are screen animations, not motor commands.</p>
 <label><input id="avatar-actions-enabled" type="checkbox" checked> Allow expressive actions</label>
 <label><input id="avatar-idle-enabled" type="checkbox" checked> Play when bored · silly faces and an accelerating carousel</label>
 <label for="avatar-play-intensity">Silliness intensity <output id="avatar-play-intensity-value">100%</output></label>
 <input id="avatar-play-intensity" type="range" min="0" max="100" step="5" value="100" aria-describedby="avatar-play-intensity-help">
 <p id="avatar-play-intensity-help">0: no idle antics. Low: subtle faces and one gentle turn. High: exaggerated faces, up to three turns and enthusiastic cheers. Does not change conversation or gaze tracking.</p>
 <label for="avatar-idle-delay">Bored after <output id="avatar-idle-delay-value" for="avatar-idle-delay">20 seconds</output> of quiet</label>
 <input id="avatar-idle-delay" type="range" min="5" max="180" step="5" value="20" aria-describedby="avatar-idle-delay-help">
 <p id="avatar-idle-delay-help">5 seconds–3 minutes. Default: 20 seconds. Changes apply immediately and are saved. The same quiet pause separates play sequences. Invitations to pet Robby stay rare.</p>
 <label><input id="avatar-play-voice-enabled" type="checkbox" checked> Cheer during carousel play · LLM-generated Hui / Whee / Woohoo</label>
 <p>One to three turns depending on intensity: slowly at first, then faster and more delighted, a gentle brake and six seconds of happy afterglow. Optional cheers start with actual audio playback; at most once per 45 seconds. Screen-only, never a motor command.</p>
 <label><input id="avatar-petting-enabled" type="checkbox" checked> Stroke Robby’s head with the left mouse button</label>
 <label for="avatar-cuddle-intensity">Cuddly personality <output id="avatar-cuddle-intensity-value">100%</output></label>
 <input id="avatar-cuddle-intensity" type="range" min="0" max="100" step="5" value="100" aria-describedby="avatar-cuddle-intensity-help">
 <p id="avatar-cuddle-intensity-help">Controls the softness of petting expressions, warmth of spoken reactions and frequency of optional invitations. 0: no petting reactions or invitations. High: more expressive, never demanding. Saved on this device.</p>
 <label><input id="avatar-pet-speech-enabled" type="checkbox" checked> Soft vocalizations and short LLM-generated reactions while stroking</label>
 <p>Robby can hum, chuckle softly or say a few warm words while you stroke his head. Depending on intensity, spoken reactions are spaced 12–36 seconds apart. He waits while you speak or another reply plays.</p>
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
  ['Carousel · accelerating joy + afterglow',[{kind:'carousel',direction:'clockwise',duration_s:6},{kind:'afterglow',duration_s:6}]],
  ['Contented kitten-like smile',{kind:'pet',duration_s:3}],['Soft upward puppy eyes',{kind:'puppy',duration_s:4}]];
 const select=panel.querySelector('select');items.forEach(([label],i)=>select.add(new Option(label,i)));
 const delay=document.getElementById('avatar-idle-delay'),delayValue=document.getElementById('avatar-idle-delay-value');
 try{const stored=localStorage.getItem('robby.idleDelaySeconds'),value=stored===null?20:Number(stored);
  if(Number.isFinite(value)&&value>=5&&value<=180)delay.value=String(value);
 }catch{}
 function applyDelay(){actions.setIdleDelay(Number(delay.value));delayValue.value=`${actions.idleDelayMs/1000} seconds`;delay.setAttribute('aria-valuetext',delayValue.value);}
 applyDelay();delay.oninput=()=>{applyDelay();try{localStorage.setItem('robby.idleDelaySeconds',String(actions.idleDelayMs/1000));}catch{}};
 for(const [id,key] of [['avatar-play-intensity','playIntensity'],['avatar-cuddle-intensity','cuddleIntensity']]){
  const slider=document.getElementById(id),out=document.getElementById(id+'-value');
  try{const raw=localStorage.getItem('robby.'+key),n=raw===null?1:Number(raw);if(Number.isFinite(n)&&n>=0&&n<=1)slider.value=String(n*100);}catch{}
  const apply=()=>{actions.setIntensity(key,Number(slider.value)/100);out.value=`${Math.round(actions[key]*100)}%`;slider.setAttribute('aria-valuetext',out.value);};
  apply();slider.oninput=()=>{apply();actions.clear();try{localStorage.setItem('robby.'+key,String(actions[key]));}catch{}};
 }
 for(const [id,key] of [['avatar-actions-enabled','enabled'],['avatar-idle-enabled','idleEnabled'],['avatar-play-voice-enabled','playVoiceEnabled'],['avatar-listening-enabled','listeningEnabled'],['avatar-petting-enabled','pettingEnabled'],['avatar-pet-speech-enabled','petSpeechEnabled'],['avatar-pet-invites-enabled','petInvitesEnabled']]){
  const box=document.getElementById(id);try{box.checked=localStorage.getItem('robby.'+id)!=='false';}catch{}actions[key]=box.checked;
  box.onchange=()=>{actions[key]=box.checked;actions.warning='';actions.clear();try{localStorage.setItem('robby.'+id,String(box.checked));}catch{}};
 }
 document.getElementById('avatar-perform').onclick=()=>{const value=items[+select.value][1];actions.play((Array.isArray(value)?value:[value]).map(s=>({duration_s:2,...s})));};
 document.getElementById('avatar-stop').onclick=()=>actions.clear();
 let last='';const timer=setInterval(()=>{const text=actions.warning|| (actions.active?`Performing ${actions.active.kind.replaceAll('_',' ')} · ${actions.active.source}`:!actions.enabled?'Expressive actions disabled.':'Camera tracking resumes automatically after each trick.');if(text!==last){document.getElementById('avatar-action-status').textContent=text;last=text;}const notice=document.getElementById('action-notice');if(notice){notice.hidden=!actions.warning;notice.textContent=actions.warning||'';}},250);
 addEventListener('pagehide',()=>clearInterval(timer),{once:true});
 return panel;
}
