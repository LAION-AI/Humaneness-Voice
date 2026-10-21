// Local perceptual calibration. Stored values are geometry, never camera frames.
export function attachEyeContactControls(rig,tracker,ensureCamera){
 const panel=document.createElement('details');panel.className='desk-controls gaze-controls';
 panel.innerHTML=`<summary>Eye contact · tracking and calibration <span>↗</span></summary>
 <p>Robby follows the midpoint of your eyes, not your head tilt, nose or facial rotation. Eye spacing provides a rough distance cue. Tracking is local; it is not eye-gaze or identity recognition.</p>
 <div class="desk-buttons"><button id="gaze-start" type="button">Calibrate eye contact</button><button id="gaze-finish" type="button" disabled>Finish</button></div>
 <p>Move your whole head left/right, up/down and closer/farther. Hold each position briefly. <b>Y</b>: looks right, save this example. <b>X</b>: freeze this observation and face, then adjust with arrow keys (Shift = 5°). <b>Y</b> saves the corrected pose. Escape cancels a correction. Use several different positions, ideally 8–12. Shortcuts work only during calibration and never inside text fields.</p>
 <div class="desk-buttons"><button id="gaze-yes" type="button" disabled>Y · looks at me</button><button id="gaze-no" type="button" disabled>X · freeze and correct</button><button id="gaze-left" type="button" disabled aria-label="Calibration look left">←</button><button id="gaze-up" type="button" disabled aria-label="Calibration look up">↑</button><button id="gaze-down" type="button" disabled aria-label="Calibration look down">↓</button><button id="gaze-right" type="button" disabled aria-label="Calibration look right">→</button></div>
 <p id="gaze-status" role="status"></p><div class="desk-buttons"><button id="gaze-undo" type="button">Undo last example</button><button id="gaze-reset" type="button">Reset eye calibration</button></div>
 <p>Only numeric eye positions, approximate distances and your accepted angles stay in this browser. No images are stored or uploaded. Reset after changing webcam or screen placement. This is a subjective visual calibration, not a physical robot calibration.</p>`;
 document.querySelector('.desk-controls').before(panel);
 const $=id=>document.getElementById(id),state={active:false,frozen:null,step:0,done:false,loading:false,error:'',loadId:0};let previousManual=null,lastStatus='';
 const steps=['Sit normally and look at Robby.','Move your whole head a little to your left. Do not just tilt it.',
  'Move your whole head to your right.','Move your head a little higher.','Move your head a little lower.',
  'Come a little closer to the camera.','Move a little farther from the camera.','Try one diagonal position of your choice.'];
 const tour=document.createElement('section');tour.className='gaze-tour';tour.id='gaze-tour';tour.hidden=true;
 tour.setAttribute('role','region');tour.setAttribute('aria-label','Eye contact calibration tutorial');tour.setAttribute('aria-live','polite');
 tour.innerHTML='<div class="gaze-tour-step"></div><h2></h2><p class="gaze-tour-instruction"></p><p class="gaze-tour-keys"></p><p class="gaze-tour-feedback"></p><p class="gaze-tour-privacy">Local calibration · no images saved</p>';document.body.append(tour);
 const showTour=()=>{
  tour.hidden=!(state.active||state.done||state.loading);if(tour.hidden)return;
  if(state.loading||state.error){
   tour.querySelector('.gaze-tour-step').textContent='EYE CONTACT · CAMERA';
   tour.querySelector('h2').textContent=state.error?'Camera access needed':'Allow the camera to begin';
   tour.querySelector('.gaze-tour-instruction').textContent=state.error||'Choose Allow in the browser permission prompt. Robby is facing forward; video stays on this device.';
   tour.querySelector('.gaze-tour-keys').textContent=state.loading?'Esc · cancel setup':'Enter / Esc · close';
   tour.querySelector('.gaze-tour-feedback').textContent=lastStatus;return;
  }
  tour.querySelector('.gaze-tour-step').textContent=state.done?'CALIBRATION SAVED':`EYE CONTACT · ${state.step+1} / ${steps.length}`;
  tour.querySelector('h2').textContent=state.done?'Ready. Robby will use your corrections.':state.frozen?'Fine-tune this frozen position':steps[state.step];
  tour.querySelector('.gaze-tour-instruction').textContent=state.done?'Try moving naturally. You can recalibrate any time.':state.frozen?'The eye sample and facial animation are frozen. Adjust until Robby looks at you.':'Hold this position briefly. Does Robby look into your eyes?';
  tour.querySelector('.gaze-tour-keys').textContent=state.done?'Enter / Esc · close':state.frozen?'← → pan · ↑ ↓ tilt · Shift = 5° · Y save & next · Esc cancel':'Y · yes, save & next    X · no, freeze & correct    Esc · finish';
  tour.querySelector('.gaze-tour-feedback').textContent=lastStatus;
 };
 const status=text=>{lastStatus=text;$('gaze-status').textContent=text;showTour();};
 const count=()=>`${tracker.samples.length} saved position${tracker.samples.length===1?'':'s'}`;
 const refresh=()=>{for(const id of ['gaze-yes','gaze-no','gaze-finish'])$(id).disabled=!state.active;
  for(const id of ['gaze-left','gaze-right','gaze-up','gaze-down'])$(id).disabled=!state.frozen;
  $('gaze-start').disabled=state.active;
 };
 function restore(){state.frozen=null;rig.setManual(previousManual?.pan||0,previousManual?.tilt||0,!state.active&&!!previousManual?.on);refresh();}
 async function start(){
  if(state.active||state.loading)return;
  previousManual={on:rig.state.manual,pan:rig.state.manualPan,tilt:rig.state.manualTilt};
  rig.center();rig.setExploded(0);rig.state.pan=rig.state.tilt=0;rig.setManual(0,0);
  state.loading=true;state.done=false;state.error='';const ticket=++state.loadId;
  document.getElementById('avatar').scrollIntoView({block:'center',behavior:'smooth'});
  status('Starting local camera tracking…');const available=await ensureCamera();
  if(ticket!==state.loadId)return;
  state.loading=false;
  if(!available){state.done=true;state.error='Allow camera access in your browser, then start calibration again. No microphone is required.';restore();status('Camera unavailable; no calibration started.');return;}
  rig.center();rig.setExploded(0);rig.state.pan=rig.state.tilt=0;rig.state.gazeCalibration=true;state.active=true;state.step=0;state.done=false;panel.open=true;refresh();
  document.getElementById('avatar').scrollIntoView({block:'center',behavior:'smooth'});
  status('Calibration active. Move to a position, hold, then Y to confirm or X to freeze/correct. '+count()+'.');
 }
 function finish(completed=false){state.active=false;state.done=completed;restore();rig.setManual(0,0,false);rig.state.gazeCalibration=false;refresh();status('Live eye-position following resumed. '+count()+'.');}
 function freeze(){
  if(!state.active||state.frozen)return;
  if(!tracker.latest||performance.now()-tracker.at>500){status('No fresh eyes detected. Face the camera, then try X again.');return;}
  const s=rig.snapshot();state.frozen={observation:{...tracker.latest},pan:s.pan,tilt:s.tilt,time:performance.now(),
   input:{...s.controls,pose:structuredClone(s.displayPose||s.pose),freezeExpression:true}};
  rig.setManual(s.pan,s.tilt);refresh();status('Frozen observation and face. Arrow keys adjust pan/tilt; Y saves; Escape cancels.');
 }
 function accept(){
  if(!state.active)return;
  if(!state.frozen&&(!tracker.latest||performance.now()-tracker.at>500)){status('No fresh eyes detected; no calibration sample saved.');return;}
  const s=rig.snapshot(),o=state.frozen?.observation||tracker.latest;
  tracker.accept(o,s.pan,s.tilt);restore();state.step++;
  if(state.step>=steps.length){finish(true);return;}
  status('Accepted. '+count()+'. Move to the next position shown above.');
 }
 function nudge(key,step=1){
  const f=state.frozen;if(!f)return;
  if(key==='ArrowLeft')f.pan=Math.max(-65,f.pan-step);if(key==='ArrowRight')f.pan=Math.min(65,f.pan+step);
  if(key==='ArrowUp')f.tilt=Math.min(15,f.tilt+step);if(key==='ArrowDown')f.tilt=Math.max(-15,f.tilt-step);
  rig.setManual(f.pan,f.tilt);status(`Frozen · pan ${f.pan.toFixed(1)}° · tilt ${f.tilt.toFixed(1)}°. Y saves; Escape cancels.`);
 }
 $('gaze-start').onclick=start;$('gaze-finish').onclick=()=>finish();$('gaze-no').onclick=freeze;$('gaze-yes').onclick=accept;
 for(const [id,key] of [['left','ArrowLeft'],['right','ArrowRight'],['up','ArrowUp'],['down','ArrowDown']])$('gaze-'+id).onclick=()=>nudge(key);
 $('gaze-undo').onclick=()=>{tracker.undo();status('Removed the last saved position. '+count()+'.');};
 $('gaze-reset').onclick=()=>{if(state.frozen)restore();tracker.clear();status('Calibration cleared. Using the uncalibrated eye-position heuristic.');};
 const keydown=e=>{
  if(state.loading&&e.key==='Escape'){state.loadId++;state.loading=false;restore();showTour();e.preventDefault();return;}
  if(state.done&&['Escape','Enter'].includes(e.key)){state.done=false;state.error='';showTour();e.preventDefault();return;}
  if(!state.active||e.ctrlKey||e.metaKey||e.altKey||(e.repeat&&!e.key.startsWith('Arrow')))return;
  if(e.target.closest?.('textarea,select,input:not([type="range"]):not([type="checkbox"]),[contenteditable="true"]'))return;
  const key=e.key.toLowerCase();
  if(!['x','y','escape','arrowleft','arrowright','arrowup','arrowdown'].includes(key))return;
  e.preventDefault();e.stopImmediatePropagation();
  if(key==='y')accept();else if(key==='x')freeze();else if(key==='escape'){if(state.frozen){restore();status('Correction cancelled. '+count()+'.');}else finish();}
  else nudge(e.key,e.shiftKey?5:1);
 };
 const faceCheck=setInterval(()=>{if(state.active&&!state.frozen){
  if(performance.now()-tracker.at>1300){if(!lastStatus.startsWith('Face not found'))status('Face not found. Make both eyes visible to the camera. No sample will be saved yet.');}
  else if(lastStatus.startsWith('Face not found'))status('Eyes found. Hold this position, then Y or X.');
 }},800);
 addEventListener('keydown',keydown,true);addEventListener('pagehide',()=>{removeEventListener('keydown',keydown,true);clearInterval(faceCheck);},{once:true});
 status('Eye-position heuristic ready. '+count()+'. Optional calibration can learn camera/screen offsets.');refresh();
 return {state,start,finish,freeze,accept,nudge,status:()=>lastStatus};
}
