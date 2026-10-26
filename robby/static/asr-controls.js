// Model loading never blocks the camera or microphone capture loop.
export class ASRControls{
 constructor(client){
  this.client=client;this.select=document.getElementById('asr-model');this.status=document.getElementById('asr-model-status');this.cache=document.getElementById('asr-cache-status');
  this.desired='parakeet_v3';this.active='nemotron';this.connected=false;
  try{
   let saved=localStorage.getItem('robby.asrModel');
   if(localStorage.getItem('robby.asrDefaultVersion')!=='2'){
    if(!saved||saved==='nemotron'){saved='parakeet_v3';localStorage.setItem('robby.asrModel',saved);}
    localStorage.setItem('robby.asrDefaultVersion','2');
   }
   if([...this.select.options].some(o=>o.value===saved))this.desired=saved;
  }catch{}
  this.select.value=this.desired;
  this.select.onchange=async()=>{
   this.desired=this.select.value;try{localStorage.setItem('robby.asrModel',this.desired);}catch{}
   this.status.textContent='Loading the selected recognizer in the background…';
   try{await this.client.start(false);this.client.send('asr:model:'+this.desired);}catch(e){this.status.textContent=e.message;}
  };
  this.refresh();this.timer=setInterval(()=>{if(!document.hidden)this.refresh();},5000);
  addEventListener('pagehide',()=>clearInterval(this.timer),{once:true});
 }
 async refresh(){
  if(this.refreshing)return;this.refreshing=true;
  try{
   const r=await fetch('/api/asr/models');if(!r.ok)throw Error('Model status unavailable');const data=await r.json();
   this.cache.textContent=data.models.map(m=>`${m.label}: ${m.phase.replaceAll('_',' ')}${m.error?' · '+m.error:''}`).join(' · ');
   for(const option of this.select.options){const row=data.models.find(m=>m.id===option.value);option.disabled=!!row&&!row.ready;}
  }catch{this.cache.textContent='Model worker unavailable. Original Nemotron remains available.';}
  finally{this.refreshing=false;}
 }
 message(m){
  if(m.type==='ready'){
   this.connected=true;
   if(m.asr_settings?.selected_model!==this.desired&&m.asr_settings?.loading_model!==this.desired)this.client.send('asr:model:'+this.desired);
   if(m.asr_settings)this.message(m.asr_settings);
  }
  if(m.type==='asr_settings'){
   this.active=m.selected_model||'nemotron';
   const name=[...this.select.options].find(o=>o.value===this.active)?.textContent||this.active;
   this.status.textContent=m.loading_model?'Loading '+this.select.selectedOptions[0].textContent+' in the background… Current recognizer: '+name:
    `Final transcript: ${name}. Live preview and interruptions: Nemotron. Replies: German by default.`+(m.switch_ms==null?'':` Model switch: ${Math.round(m.switch_ms)} ms.`);
   if(!m.loading_model&&this.active===this.desired)this.select.value=this.active;
   this.refresh();
  }
  if(m.type==='asr_finalizing')this.status.textContent='Speech pause detected · '+(this.select.selectedOptions[0]?.textContent||m.model)+' is transcribing the completed segment…';
  if(m.type==='asr_sentence'&&m.asr)this.status.textContent=`Final transcript from ${m.asr.model}${m.asr.transcribe_ms==null?'':` · ${Math.round(m.asr.transcribe_ms)} ms`}. Replies: German by default.`;
 }
}
