// Only model/provider preferences persist. No API key enters browser storage.
export class LLMControls{
 constructor(client){
  this.client=client;this.model=document.getElementById('llm-model');this.provider=document.getElementById('llm-provider');this.status=document.getElementById('llm-status');
  this.desired={model:'local',provider:'local'};this.current=null;this.connected=false;this.hasSaved=false;
  try{const saved=JSON.parse(localStorage.getItem('robby.llmSelection'));if(saved&&['qwen','local','gemma12'].includes(saved.model)){this.desired=saved;this.hasSaved=true;}}catch{}
  this.model.onchange=()=>{this.desired={model:this.model.value,provider:this.model.value==='qwen'?'cerebras/fp16':'local'};this.render();this.apply();};
  this.provider.onchange=()=>{this.desired.provider=this.provider.value;this.apply();};
  document.getElementById('llm-key-use').onclick=async()=>{
   const input=document.getElementById('llm-personal-key'),value=input.value.trim();input.value='';
   if(!value)return;
   try{await client.start(false);client.send('llm:key:'+value);}catch(e){this.status.textContent=e.message;}
  };
  document.getElementById('llm-key-clear').onclick=()=>{document.getElementById('llm-personal-key').value='';client.send('llm:key:clear');};
  this.load();
 }
 async load(){
  try{
   const r=await fetch('/api/llm/models',{cache:'no-store'});if(!r.ok)throw Error('Language model catalogue unavailable');this.catalog=await r.json();
   this.model.replaceChildren(...this.catalog.models.map(m=>new Option(m.label,m.id)));
   if(!this.hasSaved)this.desired=this.catalog.default;
   if(!this.catalog.models.some(m=>m.id===this.desired.model))this.desired=this.catalog.default;
   if(this.desired.model==='qwen'&&!this.catalog.providers.some(p=>p.id===this.desired.provider))this.desired.provider='cerebras/fp16';
   this.render();if(this.connected)this.client.send('llm:select:'+JSON.stringify(this.desired));
   this.status.textContent=this.catalog.server_key_available?'The server key is configured and hidden. Choose a model and endpoint.':'OpenRouter needs a personal key or an administrator-configured server key. Local Nemotron needs no API key.';
  }catch(e){this.status.textContent=e.message;}
 }
 render(){
  this.model.value=this.desired.model;
  const rows=this.desired.model!=='qwen'?[{id:'local',label:'Local GPU'}]:this.catalog?.providers||[{id:'cerebras/fp16',label:'Cerebras · FP16'}];
  this.provider.replaceChildren(...rows.map(p=>new Option(p.label,p.id)));this.provider.value=this.desired.provider;this.provider.disabled=this.desired.model!=='qwen';
 }
 async apply(){
  this.hasSaved=true;
  try{localStorage.setItem('robby.llmSelection',JSON.stringify(this.desired));}catch{}
  this.status.textContent='Applying language model for the next response…';
  try{await this.client.start(false);this.client.send('llm:select:'+JSON.stringify(this.desired));}catch(e){this.status.textContent=e.message;}
 }
 disconnected(){this.connected=false;this.current=null;document.getElementById('llm-personal-key').value='';this.status.textContent='Conversation ended. Model choice is remembered; personal keys are discarded.';}
 message(m){
  if(m.type==='ready'){
   this.connected=true;this.client.send('llm:select:'+JSON.stringify(this.desired));
   if(m.llm_settings)this.message(m.llm_settings);
  }
  if(m.type==='llm_settings'){
   this.current={model:m.model,provider:m.provider};
   const local=m.model!=='qwen';
   this.status.textContent=(local?(m.model==='gemma12'?'Gemma 4 12B Q4 on this GPU':'Local Nemotron on this GPU'):`Qwen 3.8 27B · ${m.provider}`)+' · '+(local?'no API key needed':m.key_mode==='personal_session'?'personal key for this connection':m.key_available?'hidden server key':'API key missing')+'. Changes apply to the next response.';
   document.querySelector('.privacy').textContent=local?'Camera tracking stays in this browser. Speech recognition, voice matching, local LLM and TTS run on this server. No microphone recordings are saved.':'Camera tracking stays in this browser. Audio is processed on this server; conversation text is sent to OpenRouter and the selected provider. The server API key stays hidden.';
  }
 }
}
