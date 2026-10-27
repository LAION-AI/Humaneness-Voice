// Optional browser-owned prototypes. Metadata only is visible by default;
// vectors are requested solely after explicit permission to retain profiles.
export class SocialProfiles {
 constructor(client){
  this.client=client;this.enabled=false;this.saved=[];this.approved=new Set();this.lastExport=0;
  try{const d=JSON.parse(localStorage.getItem('robby.consentedVoiceProfiles.v1')||'null');if(d?.consent===true&&Array.isArray(d.profiles)){this.saved=d.profiles.slice(0,8);this.enabled=true;this.approved=new Set(this.saved.map(x=>x.person_id));}}catch{}
  const panel=document.createElement('details');panel.className='settings';panel.id='social-profiles';
  panel.innerHTML='<summary>People Robby knows · voice profiles</summary><p>Names come from introductions. A person can have several voice prototypes, so different intonation need not create another person. Vocal traits are uncertain impressions, not actual age or gender. Facts and interests use the separate optional wiki memory.</p><div id="social-profile-list"></div><label><input type="checkbox" id="remember-voice-profiles"> Remember these voice profiles on this browser, with everyone’s permission</label><p id="social-profile-storage" role="status"></p><button type="button" id="forget-voice-profiles">Forget voice profiles</button><p>Camera and webcam Face ID remain independently optional. Enable the camera and “Recognize people by webcam Face ID” to connect a visible speaking face to its voice; face VAD does not need to block conversation.</p>';
  document.querySelector('aside').append(panel);this.list=panel.querySelector('#social-profile-list');this.status=panel.querySelector('#social-profile-storage');this.box=panel.querySelector('input');this.box.checked=this.enabled;
  const docs=document.createElement('p');docs.innerHTML='<a href="/static/social-identity-report.html" target="_blank" rel="noopener">How identity works · tests, defaults and limitations</a>';panel.append(docs);
  this.status.textContent=this.enabled?'Consented profiles will be restored on connection.':'Session only · no voice profiles saved.';
  this.box.onchange=()=>{if(this.box.checked&&!confirm('Save the currently known people’s names, voice embeddings and uncertain vocal traits on this browser? Continue only with their permission. These are biometric clues, not authentication. The browser storage is not encrypted.'))this.box.checked=false;
   this.enabled=this.box.checked;this.approveNext=this.enabled;
   if(this.enabled)this.command('export');else{this.saved=[];this.approved.clear();try{localStorage.removeItem('robby.consentedVoiceProfiles.v1');}catch{}this.status.textContent='Session only · saved voice profiles removed.';}
  };
  const reset=()=>{this.enabled=false;this.box.checked=false;this.saved=[];this.approved.clear();try{localStorage.removeItem('robby.consentedVoiceProfiles.v1');}catch{}this.status.textContent='Face and voice profiles forgotten. Separate factual wiki memory is unchanged.';};
  window.addEventListener('robby-forget-people',reset);
  const forget=panel.querySelector('button');forget.textContent='Forget all face + voice profiles';
  forget.onclick=()=>{reset();document.getElementById('forget-multimodal-people')?.click();this.command('forget');};
 }
 command(action){this.client.send('profiles:'+JSON.stringify({action,consent:this.enabled,profiles:action==='import'?this.saved:undefined}));}
 message(m){
  if(m.type==='ready'&&this.enabled&&this.saved.length)this.command('import');
  if(m.type==='identity'&&m.sentence_mode){
   const fragment=document.createDocumentFragment();
   for(const u of m.speakers||[]){const row=document.createElement('article');row.className='person-card';const heading=document.createElement('strong');heading.textContent=u.name;row.append(heading);
    const p=document.createElement('p');p.textContent=`${u.voice_variants||0} acoustic prototype(s) · ${u.named?'name supplied':'name not supplied'}`;row.append(p);
    const hint=document.createElement('p');hint.textContent=u.social_profile?'Possibly '+u.social_profile.traits.map(t=>t.description).join(', ')+` · ${u.social_profile.observations} accepted voice observation(s).`:'No current vocal-trait summary.';row.append(hint);fragment.append(row);}
   if(!fragment.childNodes.length){const p=document.createElement('p');p.textContent='No speaker profile yet. Start talking; introductions are optional.';fragment.append(p);}this.list.replaceChildren(fragment);
   if(this.enabled&&performance.now()-this.lastExport>800){this.lastExport=performance.now();this.command('export');}
  }
  if(m.type==='speaker_profiles'&&this.enabled){
   const candidates=(m.profiles||[]).slice(0,8);
   if(this.approveNext){this.approved=new Set(candidates.map(p=>p.person_id));this.approveNext=false;}
   const allowed=candidates.filter(p=>this.approved.has(p.person_id));
   this.saved=allowed;
   try{localStorage.setItem('robby.consentedVoiceProfiles.v1',JSON.stringify({consent:true,profiles:allowed}));this.status.textContent=`${allowed.length} consented profile(s) saved on this browser.`+(candidates.length>allowed.length?' New people remain session-only; enable again with their permission to include them.':'');}
   catch{this.enabled=false;this.box.checked=false;this.status.textContent='Browser storage unavailable; session-only profiles.';}
  }
 }
}
