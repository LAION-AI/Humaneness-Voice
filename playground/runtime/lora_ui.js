/* Full-FT base by default; DPO and Ref15 remain explicit, experimental opt-ins. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const defaultId = 'subset_pitch_quality_p2_rank016';
  const base = () => ({id:null,weight:1,ref15:false,ref15_weight:1});
  const recommended = () => ({...base(),id:data?.default??null});
  let saved=base(),draft,data;
  const copy = value => ({id:value.id,weight:value.weight,ref15:!!value.ref15,ref15_weight:value.ref15_weight??1});
  function label(pick) {
    const parts=[];
    if(pick.id&&pick.weight>0)parts.push(`${data?.adapters.find(x=>x.id===pick.id)?.label||pick.id} · Rank ${Number(pick.id.split('rank').pop())} · ${pick.weight.toFixed(2)}×`);
    if(pick.ref15&&pick.ref15_weight>0)parts.push(`Ref15 · Rank 128 · ${pick.ref15_weight.toFixed(2)}×`);
    return parts.length?parts.join(' + '):(globalThis.actingLabel||data?.base_label||(data?.base_stage==='S3'?'Original S3':'Ref15 Full FT · Step 26.774'))+' · no LoRA';
  }
  function save() { try {localStorage.setItem('small.adapters.v4', JSON.stringify(saved));} catch {} }
  function summary() {return label(saved);}
  function refreshSummary() {
    $('lora-summary').textContent=summary();globalThis.updateConfigurationSummary?.();
    for(const id of ['lora-open','lora-open-sidebar'])$(id).disabled=!data?.enabled;
    for(const b of document.querySelectorAll('[data-lora-preset]')){
      b.disabled=!data?.enabled;
      const active=(!saved.id||saved.weight===0?'':'dpo')+(saved.ref15&&saved.ref15_weight>0?'ref15':'');
      b.setAttribute('aria-pressed',String(active==={base:'',dpo:'dpo',ref15:'ref15',both:'dporef15'}[b.dataset.loraPreset]));
    }
  }
  for(const b of document.querySelectorAll('[data-lora-preset]'))b.onclick=()=>{
    if(!data?.enabled)return;saved=base();
    if(['dpo','both'].includes(b.dataset.loraPreset))saved.id=data.preferred_dpo||defaultId;
    if(['ref15','both'].includes(b.dataset.loraPreset))saved.ref15=true;
    save();refreshSummary();
  };
  function valid(){return ['dpo-weight','ref15-weight'].every(id=>$(id).disabled||($(id).value!==''&&$(id).checkValidity()));}
  function render() {
    $('lora-list').replaceChildren();
    const filter = $('lora-search').value.toLowerCase();
    const rows = [{id: null, label: 'No DPO · loaded base model'}, ...(data?.adapters || [])];
    for (const row of rows.filter(x => `${x.label} ${x.id || ''} Rank ${x.rank || ''}`.toLowerCase().includes(filter))) {
      const option = document.createElement('label'); option.className = 'lora-option';
      const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'dpo-adapter';
      radio.value = row.id || ''; radio.checked = draft.id === row.id;
      const text = document.createElement('span');text.textContent = row.label;
      const meta = document.createElement('small');meta.textContent = row.id ?
        `Rank ${row.rank}${row.id===(data?.preferred_dpo||defaultId)?' · preferred optional DPO':''}\n${row.id}` : 'DPO off; independent Ref15 toggle unchanged.';
      text.append(meta); option.append(radio, text);
      radio.onchange = () => {draft.id = row.id; render();};$('lora-list').append(option);
    }
    $('lora-selection').replaceChildren();
    for(const [id,title,key,enabled] of [['dpo-weight','DPO strength','weight',!!draft.id],
        ['ref15-weight','Ref15 strength','ref15_weight',draft.ref15]]){
      const picked=document.createElement('div');picked.className='lora-picked';
      const strong=document.createElement('strong');strong.textContent=title;picked.append(strong);
      if(id==='ref15-weight'){
        const choice=document.createElement('label'),toggle=document.createElement('input');
        toggle.id='ref15-enable';toggle.type='checkbox';toggle.checked=draft.ref15;
        toggle.onchange=()=>{draft.ref15=toggle.checked;render();};
        choice.append(toggle,document.createTextNode(' Ref15 · Rank 128 enable'));picked.append(choice);
      }
      const strength=document.createElement('label');strength.textContent=' Strength ';
      const input=document.createElement('input');input.id=id;input.type='number';
      input.min='0';input.max='2';input.step='0.05';input.value=String(draft[key]);input.disabled=!enabled;
      input.oninput=()=>{if(input.checkValidity()&&input.value!=='')draft[key]=Number(input.value);
        $('lora-apply').disabled=!data?.enabled||!valid();};
      strength.append(input);picked.append(strength);$('lora-selection').append(picked);
    }
    $('lora-warning').textContent=
      `Default: loaded base model without LoRA; up to ${globalThis.smallReferenceSeconds||15} s reference context. `+
      (data?.cross_base&&(draft.id||draft.ref15)?'These adapters were trained on original S3; mixing with Full FT is experimental. ':'')+
      (draft.ref15?'Ref15 requires reference audio; trained up to 14.96 seconds. DPO + Ref15 and 20-second references are experimental.':'');
    $('lora-apply').disabled=!data?.enabled||!valid();
  }
  function open() {draft = copy(saved);$('lora-search').value = '';render();$('lora-dialog').showModal();}
  $('lora-open').onclick = open;$('lora-open-sidebar').onclick = open;
  for(const id of ['lora-close','lora-cancel'])$(id).onclick = () => $('lora-dialog').close();
  $('lora-clear').onclick = () => {draft=base();render();};
  $('lora-default').onclick = () => {draft=recommended();render();};
  $('lora-search').oninput = render;
  $('lora-apply').onclick = () => {if(!data?.enabled||!valid())return;
    saved=copy(draft);save();refreshSummary();$('lora-dialog').close();};
  globalThis.smallLoras = {selection: () => data?.enabled?copy(saved):base(), summary,
    modelChanged() {if(!data?.enabled)saved=base();refreshSummary();},
    resetDefault() {saved=recommended();save();refreshSummary();},
    result(box, report) {const line=document.createElement('div');line.className='lora-result';
      line.textContent = `Model: ${label(report.selection)} · switch ${report.switch_ms || 0} ms${report.experimental_mix?' · experimental combination':''}`;box.append(line);}};
  refreshSummary();
  fetch('/api/adapters/catalog').then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);return r.json();}).then(value=>{
    // New full-FT defaults do not resurrect the former automatic DPO in v3.
    data=value;saved=data.enabled?recommended():base();try{const pick=JSON.parse(localStorage.getItem('small.adapters.v4'));if(data.enabled && pick &&
      (pick.id===null || data.adapters.some(x=>x.id===pick.id)) && Number.isFinite(pick.weight) && pick.weight>=0 && pick.weight<=2 &&
      typeof pick.ref15==='boolean'&&Number.isFinite(pick.ref15_weight)&&pick.ref15_weight>=0&&pick.ref15_weight<=2)saved=copy(pick);}catch{}
    $('lora-catalog-status').textContent=`${data.adapters.length} optional DPO adapters + Ref15 Rank 128 · Default: no LoRA${data.cross_base?' · S3 adapters on Full FT are experimental':''}`;
    refreshSummary();if($('lora-dialog').open)render();
  }).catch(error=>{$('lora-catalog-status').textContent='Catalog unavailable: '+error.message;});
})();
