// Move existing controls without replacing their listeners, IDs or behavior.
export function organisePlayground(){
 const $=id=>document.getElementById(id),aside=document.querySelector('aside');
 const recognition=document.createElement('details');recognition.className='settings recognition-settings';recognition.id='recognition-settings';
 const summary=document.createElement('summary');summary.textContent='Recognition & privacy settings';recognition.append(summary,$('speaker-controls'));aside.append(recognition);
 const advanced=document.querySelector('details.settings:not(.recognition-settings)');
 for(const node of [document.querySelector('.expression-controls'),document.querySelector('.desk-controls:not(.gaze-controls)'),document.querySelector('.gaze-controls')])if(node)advanced.append(node);
 const input=document.querySelector('.input-health');const health=document.createElement('details');health.className='input-details';const title=document.createElement('summary');title.textContent='Microphone & speech recognition';health.append(title,input);$('quick-controls').after(health);
}
