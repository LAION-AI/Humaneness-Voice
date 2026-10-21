// Inspection controls for the shared, CPU-rendered desk-robot conversation page.
import {profiles} from './emotions.js';
export function attachDeskControls(rig){
 const caption=document.createElement('p');caption.className='desk-caption';caption.textContent='Drag the robot to orbit · scroll to zoom · arrow keys also rotate the view. Camera tracking turns the head towards you.';
 const details=document.createElement('details');details.className='desk-controls';
 details.innerHTML=`<summary>Inspect the two-axis prototype <span>↗</span></summary>
 <p>Procedural reconstruction of your desk-robot proposal. No arms, heart, cross or mechanical roll. The camera aperture sits above the round display.</p>
 <div class="desk-buttons"><button type="button" id="desk-front">Front</button><button type="button" id="desk-threequarter">Three-quarter</button><button type="button" id="desk-rear">Rear</button><button type="button" id="desk-center">Reset view</button></div>
 <label><input id="desk-manual" type="checkbox"> Manual head position (overrides camera tracking)</label>
 <label for="desk-pan">Head pan <output id="desk-pan-value">0°</output></label><input id="desk-pan" type="range" min="-90" max="90" value="0" step="1">
 <label for="desk-tilt">Head tilt <output id="desk-tilt-value">0°</output></label><input id="desk-tilt" type="range" min="-15" max="15" value="0" step="1">
 <label><input id="desk-tour" type="checkbox"> Demonstrate pan and tilt</label>
 <label><input id="desk-grid" type="checkbox"> Show axes and ground grid</label>
 <label for="desk-explode">Assembly separation <output id="desk-explode-value">assembled</output></label><input id="desk-explode" type="range" min="0" max="1" value="0" step=".05">
 <div class="desk-buttons"><button type="button" id="desk-emotions">Preview all 40 expressions</button><button type="button" id="desk-zoom-in" aria-label="Zoom in">＋ Zoom</button><button type="button" id="desk-zoom-out" aria-label="Zoom out">− Zoom</button></div>
 <label for="desk-quality">Rendering budget</label><select id="desk-quality"><option value="30">30 fps target · standard</option><option value="20">20 fps target · lower CPU</option></select>
 <dl><div><dt>Head shell / depth</dt><dd>Ø 190 / ~110–125 mm</dd></div><div><dt>Base / total height</dt><dd>Ø 220 / ~315 mm</dd></div><div><dt>Display / active image</dt><dd>Ø 150 / Ø 127.6 mm</dd></div><div><dt>Display mounting</dt><dd>4 × M4, 75 × 75 mm</dd></div><div><dt>Conservative virtual limits</dt><dd>Pan ±90° · tilt ±15° · no roll</dd></div></dl>
 <p class="desk-caution">A visual, kinematic prototype—not manufacturing CAD, collision/torque simulation or a servo controller. Part fit, cable clearance and load limits require physical checks.</p>`;
 document.getElementById('avatar').after(caption,details);
 const $=id=>document.getElementById(id),r=x=>x*Math.PI/180;
 const manual=()=>{rig.setManual(+$('desk-pan').value,+$('desk-tilt').value,$('desk-manual').checked);$('desk-pan-value').value=$('desk-pan').value+'°';$('desk-tilt-value').value=$('desk-tilt').value+'°';};
 for(const id of ['desk-pan','desk-tilt'])$(id).oninput=()=>{$('desk-manual').checked=true;manual();};
 $('desk-manual').onchange=manual;
 $('desk-front').onclick=()=>{rig.state.view=0;rig.state.elevation=0;};
 $('desk-threequarter').onclick=()=>{rig.state.view=r(-32);rig.state.elevation=r(12);};
 $('desk-rear').onclick=()=>{rig.state.view=r(180);rig.state.elevation=r(8);};
 $('desk-center').onclick=()=>{rig.center();$('desk-pan').value=$('desk-tilt').value=0;$('desk-manual').checked=$('desk-tour').checked=false;manual();};
 $('desk-tour').onchange=()=>{rig.setTour($('desk-tour').checked);if($('desk-tour').checked){$('desk-manual').checked=false;rig.state.manual=false;}};
 $('desk-grid').onchange=()=>rig.setBlueprint($('desk-grid').checked);
 $('desk-explode').oninput=()=>{rig.setExploded(+$('desk-explode').value);$('desk-explode-value').value=rig.state.exploded?Math.round(rig.state.exploded*100)+'% separated':'assembled';};
 $('desk-quality').onchange=()=>rig.state.fps=+$('desk-quality').value;
 $('desk-zoom-in').onclick=()=>rig.state.distance=Math.max(450,rig.state.distance-60);
 $('desk-zoom-out').onclick=()=>rig.state.distance=Math.min(950,rig.state.distance+60);
 let emotionTimer=null,index=0;
 function stopPreview(){clearInterval(emotionTimer);emotionTimer=null;$('desk-emotions').textContent='Preview all 40 expressions';}
 $('expression').addEventListener('change',stopPreview);
 $('desk-emotions').onclick=()=>{
  if(emotionTimer){stopPreview();$('expression').value='auto';return;}
  $('desk-emotions').textContent='Stop expression tour';index=0;
  const advance=()=>{$('expression').value=profiles[index++%profiles.length].slug;};advance();emotionTimer=setInterval(advance,1600);
 };
 const stats=document.createElement('p');stats.id='desk-render-stats';stats.className='desk-caption';details.after(stats);
 const interval=setInterval(()=>{const s=rig.snapshot();if(s.renderCount)stats.textContent=`CPU software 3D · ${rig.state.fps} fps target · render p95 ${s.renderP95Ms.toFixed(1)} ms · pan ${s.pan.toFixed(1)}° · tilt ${s.tilt.toFixed(1)}° · roll 0°`;},800);
 addEventListener('pagehide',()=>{clearInterval(interval);stopPreview();rig.dispose();},{once:true});
}
