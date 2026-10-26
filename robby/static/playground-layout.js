// Move existing controls without replacing their listeners, IDs or behavior.
export function organisePlayground(){
 const $=id=>document.getElementById(id),aside=document.querySelector('aside');
 const recognition=document.createElement('details');recognition.className='settings recognition-settings';recognition.id='recognition-settings';
 const summary=document.createElement('summary');summary.textContent='Recognition & privacy settings';recognition.append(summary,$('speaker-controls'));aside.append(recognition);
 const advanced=document.getElementById('advanced-settings')||document.querySelector('details.settings:not(.recognition-settings):not(.chat-history)');
 for(const node of [document.querySelector('.expression-controls'),document.querySelector('.desk-controls:not(.gaze-controls)'),document.querySelector('.gaze-controls')])if(node)advanced.append(node);
 const input=document.querySelector('.input-health');const health=document.createElement('details');health.className='input-details settings';const title=document.createElement('summary');title.textContent='Microphone & speech recognition';health.append(title,input);aside.append(health);
 // Keep the conversation first. Existing controls/listeners are moved intact.
 const section=(id,label,nodes)=>{const d=document.createElement('details');d.id=id;d.className='settings option-menu';const s=document.createElement('summary');s.textContent=label;d.append(s,...nodes.filter(Boolean));aside.append(d);return d;};
 section('model-settings','Models & quick replies',[$('llm-model-controls'),$('asr-model-controls'),$('quick-reply-controls')]);
 section('camera-settings','Camera & eye contact',[$('camera'),document.querySelector('.follow-controls'),$('follow-status')]);
 const conversation=$('conversation-options');conversation.classList.add('settings','option-menu');
 const proactive=document.querySelector('.quick-interrupt-switch');if(proactive)conversation.append(proactive);
 aside.append(conversation);
 document.querySelector('.under-avatar').append(document.querySelector('.call-controls'));
 document.querySelector('.conversation-row').after($('latency-dashboard'));
 document.querySelector('.under-avatar').append($('action-notice'));
 // The focus/handover block may be added dynamically; preserve every option.
 const quick=$('quick-controls');
 if(quick.children.length)section('other-conversation-settings','Additional conversation controls',[...quick.children]);
 quick.hidden=true;
 const heading=document.createElement('h2');heading.className='options-heading';heading.textContent='Settings & details';aside.prepend(heading);
 const personality=$('avatar-action-controls');if(personality){personality.classList.add('settings');aside.append(personality);}
 aside.setAttribute('aria-label','Optional settings below the conversation');
}
