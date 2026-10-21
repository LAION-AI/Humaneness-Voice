// Local previews only. No image data is sent to the server or Luna.
export function faceCanvas(video,points,size=150){
 const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
 const ctx=canvas.getContext('2d',{willReadFrequently:true}),l=points[33],r=points[263],w=video.videoWidth,h=video.videoHeight;
 const angle=Math.atan2((r.y-l.y)*h,(r.x-l.x)*w),scale=(size*.4)/Math.max(10,Math.hypot((r.x-l.x)*w,(r.y-l.y)*h));
 ctx.fillStyle='#808080';ctx.fillRect(0,0,size,size);ctx.translate(size*.5,size*55/150);ctx.scale(scale,scale);ctx.rotate(-angle);ctx.drawImage(video,-(l.x+r.x)*w/2,-(l.y+r.y)*h/2);return canvas;
}
export class PeopleView{
 constructor(mm,host){this.mm=mm;this.host=host;this.voices=[];this.voiceScores=new Map();this.lastImages=0;this.lastRender=0;this.currentVoice=null;}
 updateVideo(video,tracks,now){if(!this.mm.settings.faceIdEnabled||now-this.lastImages<1000||!video.videoWidth)return;this.lastImages=now;for(const t of tracks){try{t.liveImage=faceCanvas(video,t.points,112).toDataURL('image/jpeg',.6);}catch{}}}
 message(m){
  if(m.type==='identity'){
   if(m.speakers)this.voices=m.speakers;
   const p=m.decision||m.prediction||{};if(p.id&&p.kind==='enrolled')this.currentVoice=p.id;
   for(const s of m.verification?.reference_scores||[])this.voiceScores.set(s.name,s.cosine);
  }
  if(m.type==='endpoint'&&m.speaker_id)this.currentVoice=m.speaker_id;
  if(m.type==='identity'||m.type==='endpoint'||m.type==='visual_person'||m.type==='visual_profiles')this.render(true);
 }
 render(force=false){const now=performance.now();if(!force&&now-this.lastRender<700)return;this.lastRender=now;
  const mm=this.mm,fragment=document.createDocumentFragment(),profiles=mm.settings.faceIdEnabled?[...mm.people.profiles.values()]:[];
  const card=(name,reference,live,voice,face)=>{
   const row=document.createElement('article');row.className='person-card';const heading=document.createElement('strong');heading.textContent=name;row.append(heading);
   const pictures=document.createElement('div');pictures.className='person-pictures';
   for(const [label,url] of [['Stored reference',reference],['Current camera',live]]){
    const figure=document.createElement('figure');if(url){const img=document.createElement('img');img.src=url;img.alt=label+' for '+name;img.width=96;img.height=96;figure.append(img);}else{const placeholder=document.createElement('span');placeholder.className='picture-placeholder';placeholder.textContent=label==='Stored reference'?'No reference image':'Not visible';figure.append(placeholder);}
    const caption=document.createElement('figcaption');caption.textContent=label;figure.append(caption);pictures.append(figure);
   }
   row.append(pictures);for(const text of [voice,face]){const p=document.createElement('p');p.textContent=text;row.append(p);}fragment.append(row);
  };
  for(const p of profiles){const t=mm.tracks.find(t=>t.personId===p.id),speaker=this.voices.find(v=>v.id===p.voice?.speaker_id||v.name===p.name),score=this.voiceScores.get(p.name);
   card(p.name,p.image,t?.liveImage,'Voice: '+(speaker?(speaker.id===this.currentVoice?'current speaker · ':'enrolled · '):'not active in this conversation · ')+(score==null?'no recent score':'cosine '+score.toFixed(2)),
    'Face: '+(t?'matched '+t.id+(t.faceDistance==null?'':' · distance '+t.faceDistance.toFixed(2))+(t.speaking?' · mouth moving':''):'reference ready; not currently matched'));
  }
  for(const v of this.voices.filter(v=>!profiles.some(p=>p.voice?.speaker_id===v.id||p.name===v.name)))card(v.name,null,null,'Voice: enrolled'+(v.id===this.currentVoice?' · current speaker':''),mm.settings.faceIdEnabled?'Face: not linked yet; introduce yourself while clearly visible.':'Webcam Face ID OFF · recognized by voice.');
  if(mm.settings.faceIdEnabled)for(const t of mm.tracks.filter(t=>!t.personId))card('Unknown face · '+t.id,null,t.liveImage,'Voice: no unambiguous link','Face: visible · '+(t.speaking?'mouth moving':'not speaking'));
  if(!fragment.childNodes.length){const empty=document.createElement('p');empty.textContent='Say “Hey Robby, I’m …” to enroll your voice. Camera tracking does not require Face ID.';fragment.append(empty);}
  this.host.replaceChildren(fragment);
 }
}
