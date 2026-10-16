// Run the actual adapter controller with DOM fixtures, not rendered Chromium.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes={},presets=[];
class Element {
  constructor(){this.children=[];this.value='';this.disabled=false;this.checked=false;this.attributes={};}
  set id(value){this._id=value;nodes[value]=this;}get id(){return this._id;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(key,value){this.attributes[key]=value;}
  checkValidity(){return Number.isFinite(Number(this.value))&&Number(this.value)>=Number(this.min)&&Number(this.value)<=Number(this.max);}
  showModal(){this.open=true;}close(){this.open=false;}
}
for(const id of ['lora-summary','lora-open','lora-open-sidebar','lora-list','lora-search',
  'lora-selection','lora-warning','lora-apply','lora-dialog','lora-close','lora-cancel',
  'lora-clear','lora-default','lora-catalog-status']){const el=new Element();el.id=id;}
for(const name of ['base','dpo','ref15','both']){const el=new Element();el.dataset={loraPreset:name};presets.push(el);}
const old={id:'subset_pitch_quality_p2_rank016',weight:1},store={'small.dpo.v1':JSON.stringify(old),
  'small.adapters.v2':JSON.stringify({id:null,weight:1,ref15:false,ref15_weight:1}),
  'small.adapters.v3':JSON.stringify({...old,ref15:false,ref15_weight:1})};
const context={console,document:{getElementById:id=>nodes[id],
    querySelectorAll:selector=>{assert.equal(selector,'[data-lora-preset]');return presets;},
    createElement:()=>new Element(),createTextNode:text=>({textContent:text})},
  localStorage:{getItem:key=>store[key]??null,setItem:(key,value)=>{store[key]=value;}},
  fetch:()=>Promise.resolve({ok:true,json:()=>Promise.resolve({enabled:true,default:null,
    base_stage:'S3-Ref15-FullFT',cross_base:true,
    preferred_dpo:'subset_pitch_quality_p2_rank016',adapters:[
      {id:'subset_pitch_quality_p2_rank016',label:'Previous DPO',rank:16},
      {id:'rebalanced_rank064',label:'Rebalanced',rank:64}]})})};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'lora_ui.js'),'utf8'),context);
const selection=()=>JSON.parse(JSON.stringify(context.smallLoras.selection()));
const payloads=[];
(async()=>{
  for(let i=0;i<8;i++)await Promise.resolve();
  assert.deepEqual(selection(),{id:null,weight:1,ref15:false,ref15_weight:1});
  assert(nodes['lora-summary'].textContent.includes('Full FT'));
  for(const [name,id,ref15] of [['base',null,false],['dpo',old.id,false],['ref15',null,true],['both',old.id,true]]){
    const button=presets.find(b=>b.dataset.loraPreset===name);button.onclick();
    const value={id,weight:1,ref15,ref15_weight:1};assert.deepEqual(selection(),value);
    assert.equal(button.attributes['aria-pressed'],'true');payloads.push(value);
    assert.deepEqual(JSON.parse(store['small.adapters.v4']),value);
  }
  nodes['lora-open'].onclick();assert(nodes['lora-dialog'].open);
  assert(nodes['lora-warning'].textContent.includes('experimental'));
  nodes['ref15-enable'].checked=false;nodes['ref15-enable'].onchange();
  nodes['lora-cancel'].onclick();assert(selection().ref15,'Cancel keeps committed combination');
  nodes['lora-open-sidebar'].onclick();nodes['ref15-weight'].value='0.25';nodes['ref15-weight'].oninput();
  nodes['dpo-weight'].value='0.5';nodes['dpo-weight'].oninput();
  nodes['lora-apply'].onclick();assert(!nodes['lora-dialog'].open);
  assert.deepEqual(selection(),{id:old.id,weight:.5,ref15:true,ref15_weight:.25});payloads.push(selection());
  nodes['lora-open'].onclick();nodes['ref15-weight'].value='3';nodes['ref15-weight'].oninput();
  assert(nodes['lora-apply'].disabled);nodes['lora-apply'].onclick();assert.equal(selection().ref15_weight,.25);
  nodes['lora-cancel'].onclick();nodes['lora-open'].onclick();nodes['lora-clear'].onclick();nodes['lora-apply'].onclick();
  assert.deepEqual(selection(),{id:null,weight:1,ref15:false,ref15_weight:1});payloads.push(selection());
  presets.find(x=>x.dataset.loraPreset==='both').onclick();nodes['lora-open'].onclick();
  nodes['lora-default'].onclick();nodes['lora-apply'].onclick();assert.equal(selection().id,null);assert(!selection().ref15);
  const box=new Element();context.smallLoras.result(box,{selection:selection(),switch_ms:12});
  assert(box.children[0].textContent.includes('Full FT'));
  const result={verification:'Actual LoRA UI JavaScript + DOM fixtures; no browser/GPU claim',presets:4,
    old_adapter_preferences_not_migrated:true,default_no_adapter:true,combination_weights_cancel_clear_validity:'passed',payloads};
  fs.writeFileSync(path.join(__dirname,'test-results/lora-controller.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exitCode=1;});
