// Private browser-scoped wiki memory. Never place its capability in URLs or logs.
export class MemoryControls {
 constructor(client){
  this.client=client;this.config={enabled:true,knowledge_budget:10000,history_budget:5000,hops:1};this.capability=null;
  try{Object.assign(this.config,JSON.parse(localStorage.getItem('robby.memorySettings')||'{}'));this.capability=localStorage.getItem('robby.memoryCapability');
   // Apply the new default once to existing browsers; subsequent choices persist.
   if(localStorage.getItem('robby.memoryDefaultVersion')!=='1'){this.config.enabled=true;localStorage.setItem('robby.memoryDefaultVersion','1');localStorage.setItem('robby.memorySettings',JSON.stringify(this.config));}
  }catch{}
  const d=document.createElement('details');d.className='settings option-menu';d.id='memory-settings';
  d.innerHTML=`<summary>Persistent memory</summary>
   <label><input id="memory-enabled" type="checkbox"> Remember knowledge between visits</label>
   <p>On by default. Robby saves a private text wiki and recent conversation for this browser, so he can remember people, relationships, interests and shared experiences. Turn it off or reset it here at any time. This feature does not save voiceprints, face images or microphone recordings.</p>
   <p>Updates run after replies using the selected language model. Cloud mode sends this text to your selected provider and costs additional model calls. Server-side files are private to this browser’s random access key; anyone using the same browser profile can see them.</p>
   <label for="memory-knowledge-budget">Knowledge context ceiling <output id="memory-knowledge-value"></output></label>
   <input id="memory-knowledge-budget" type="range" min="0" max="10000" step="500">
   <label for="memory-history-budget">Recent conversation ceiling <output id="memory-history-value"></output></label>
   <input id="memory-history-budget" type="range" min="0" max="5000" step="250">
   <label for="memory-hops">Related concepts <select id="memory-hops"><option value="0">Direct matches only</option><option value="1">One link away</option><option value="2">Up to two links away</option></select></label>
   <p>Token ceilings use conservative UTF-8 accounting, so actual model token counts are smaller. The local 4B model gets a reduced budget to fit its context. Only relevant entries are retrieved.</p>
   <p id="memory-status" role="status">Preparing memory when the conversation connects.</p>
   <div class="memory-buttons"><button id="memory-browse" type="button">Browse knowledge</button><button id="memory-reset" type="button">Reset knowledge & chat context</button></div>
   <label for="memory-query">Search saved knowledge</label><input id="memory-query" type="search" maxlength="500" placeholder="A name, interest or concept"><button id="memory-search" type="button">Search</button>
   <div id="memory-entries"></div><pre id="memory-entry" hidden></pre>`;
  document.querySelector('aside').append(d);this.root=d;this.$=id=>d.querySelector('#'+id);
  this.$('memory-enabled').checked=!!this.config.enabled;
  for(const [id,key] of [['memory-knowledge-budget','knowledge_budget'],['memory-history-budget','history_budget'],['memory-hops','hops']]){
   this.$(id).value=this.config[key];this.$(id).oninput=()=>{this.config[key]=Number(this.$(id).value);this.labels();};
   this.$(id).onchange=()=>this.configure();
  }
  this.$('memory-enabled').onchange=()=>{this.config.enabled=this.$('memory-enabled').checked;this.configure();};
  this.$('memory-browse').onclick=()=>this.command('list');this.$('memory-search').onclick=()=>this.command('search',{query:this.$('memory-query').value});
  this.$('memory-query').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();this.command('search',{query:e.target.value});}};
  this.$('memory-reset').onclick=async()=>{if(!confirm('Delete this browser’s saved wiki, all five-version histories and saved conversation context? This cannot be undone.'))return;
   document.querySelector('#interrupt')?.click();await this.command('reset');};
  this.labels();
 }
 labels(){this.$('memory-knowledge-value').textContent=this.config.knowledge_budget+' tokens max';this.$('memory-history-value').textContent=this.config.history_budget+' tokens max';}
 save(){try{localStorage.setItem('robby.memorySettings',JSON.stringify(this.config));}catch{}}
 token(){if(!this.capability){this.capability=[...crypto.getRandomValues(new Uint8Array(32))].map(x=>x.toString(16).padStart(2,'0')).join('');localStorage.setItem('robby.memoryCapability',this.capability);}return this.capability;}
 async configure(){
  this.save();
  try{if(this.config.enabled)this.token();if(!this.client.ws||this.client.ws.readyState!==1){if(!this.config.enabled){this.$('memory-status').textContent='Memory is off. Saved entries remain until you reset them.';return;}await this.client.start(false);}
   this.client.send('memory:'+JSON.stringify({action:'configure',...this.config,capability:this.capability}));
  }catch{this.$('memory-status').textContent='Could not configure private memory. Please reconnect.';}
 }
 async command(action,extra={}){
  if(!this.capability){this.$('memory-status').textContent='No saved wiki for this browser. Enable memory before saving knowledge.';return;}
  try{await this.client.start(false);this.client.send('memory:'+JSON.stringify({action,capability:this.capability,...extra}));}
  catch{this.$('memory-status').textContent='Memory is unavailable. Please reconnect.';}
 }
 message(m){
  if(m.type==='ready'){this.configure();return;}
  if(m.type==='memory_status'){
   const counts=typeof m.entries==='number'?` · ${m.entries} wiki entries`:'';
   this.$('memory-status').textContent=({off:'Memory is off. Saved entries are not retrieved or updated.',ready:'Memory is ready',updating:'Updating the wiki in the background',error:'Memory update unavailable; the conversation can continue.'}[m.state]||m.state)+counts+
    (m.context?` · retrieved ${m.context.activated.length} entries in ${m.context.retrieval_ms} ms`:'')+(m.update_ms?` · last update ${Math.round(m.update_ms)} ms`:'');
   if(m.reset){this.$('memory-entries').replaceChildren();this.$('memory-entry').hidden=true;this.$('memory-status').textContent='Knowledge, versions and model chat context have been reset.';}
  }
  if(m.type==='memory_entries'){
   const box=this.$('memory-entries');box.replaceChildren();
   if(!m.entries.length)box.textContent='No matching entries.';
   for(const entry of m.entries){const card=document.createElement('article'),button=document.createElement('button'),p=document.createElement('p'),del=document.createElement('button');
    button.type=del.type='button';button.textContent=entry.title+' · '+entry.kind;button.onclick=()=>this.command('entry',{id:entry.id});p.textContent=entry.summary;
    del.textContent='Delete entry';del.onclick=()=>{if(confirm('Delete this entry and its versions?'))this.command('delete',{id:entry.id});};card.append(button,p,del);box.append(card);}
  }
  if(m.type==='memory_entry'){const out=this.$('memory-entry');out.hidden=false;out.textContent=m.entry?JSON.stringify(m.entry,null,2):'Entry not found.';}
 }
}
