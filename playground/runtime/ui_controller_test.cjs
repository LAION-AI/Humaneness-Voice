// Exercise the actual UI functions with small DOM fixtures, not a browser.
// Requires Node >=18. This does not validate CSS rendering or GPU inference.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const script=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Parse the complete production script as well.
function source(name){
  const line=script.match(new RegExp('^function '+name+'\\([^\\n]*$', 'm'))[0];
  if(line.endsWith('}'))return line;
  return script.match(new RegExp('function '+name+'\\([\\s\\S]*?\\n\\}'))[0];
}
const nodes={};
for(const id of [...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]))nodes[id]={value:'',checked:false,disabled:false,
  textContent:'',required:false,valid:true,card:{open:false},checkValidity(){return this.valid;},
  closest(selector){return selector==='.settings-card'?this.card:{textContent:id};},focus(){this.focused=true;}};
for(const [id,value] of Object.entries({generation:'single',searchn:'8',searchchunk:'20',searchseconds:'6.6',
    n:'8',streaming:'offline',chunkframes:'10',inputmode:'chat',format:'caption',language:'auto',
    g:'1.5',temperature:'1',seed:'777',seconds:'',reference:'','voice-choice':'emolia_c1699',
    'model-choice':'small_acting','model-batch':'1',
    'speaker-method':'ecapa','speaker-weight':'.6',rewardmode:'clap',rewardshare:'.5',
    burstmode:'off',burstsource:'script',bursttiming:'words',burstmatch:'family',burstweight:'.25',bursttolerance:'.3',
    'ranking-mode':'legacy','ears-model':'medium','ears-metric':'ce','ears-weight':'.5'}))nodes[id].value=value;
nodes['ears-metric'].selectedOptions=[{textContent:'Content Enjoyment'}];
nodes.cfg.checked=true;nodes.align.checked=true;nodes.burstextra.checked=true;
const buttons=['search_winner','search_beam'].map(generation=>({dataset:{generation},attributes:{},
    setAttribute(key,value){this.attributes[key]=value;}}));
const context={console,history:[],smallEarsAvailable:true,methodNames:{single:'Ein Take',best:'Best-of-N',all:'Alle N Takes',
    search_winner:'Gewinner-Suche',search_beam:'Beam Search · Top 2'},
    rewardRows:[{dimension:'vn:AROU',objective:'target',target:4.5,weight:2},
                {dimension:'eiv:Anger',objective:'maximize',weight:1}],rewardReason:'Controller fixture',
    burstRows:[{label:'chuckle',start:.3,duration:.3}],$:id=>{assert(nodes[id],id);return nodes[id];},
    document:{querySelectorAll(selector){assert.equal(selector,'[data-generation]');return buttons;}}};
vm.createContext(context);
for(const name of ['updateConfigurationSummary','generationSettings','selectedRewardPlan','selectedBurstOptions','buildTurnRequest'])
    vm.runInContext(source(name),context);
const payloads=[];
const request=()=>JSON.parse(JSON.stringify(context.buildTurnRequest('Test controller request')));
context.generationSettings();
assert(nodes.searchsettings.disabled && !nodes.standardsettings.disabled);
assert(nodes.n.disabled && nodes.chunkframes.disabled && !nodes.seconds.disabled);
assert(nodes['search-status'].textContent.includes('settings remain visible'));
for(const mode of ['search_winner','search_beam']){
    nodes.generation.value=mode;
    for(let n=1;n<=16;n++)for(const chunk of ['10','20']){
        nodes.searchn.value=String(n);nodes.searchchunk.value=chunk;
        nodes.seconds.value='3.2';nodes.streaming.value='streaming';context.generationSettings();
        assert(!nodes.searchsettings.disabled && nodes.standardsettings.disabled);
        assert(nodes.n.disabled && nodes.seconds.disabled && nodes.chunkframes.disabled);
        const body=request();assert.equal(body.generation,mode);assert.equal(body.search_n,n);
        assert.equal(body.search_chunk_frames,Number(chunk));assert.equal(body.seconds,null);
        assert.equal(body.streaming,false);payloads.push(body);
        assert(nodes['configuration-summary'].textContent.includes(`${n} parallel candidates`));
        assert.equal(buttons.find(b=>b.dataset.generation===mode).attributes['aria-pressed'],'true');
        if(n===1)assert(nodes['beam-expansion'].textContent.includes('one path'));
        if(mode==='search_beam' && n===7)assert(nodes['beam-expansion'].textContent.includes('4 + 3'));
        if(mode==='search_beam' && n===16)assert(nodes['beam-expansion'].textContent.includes('8 + 8'));
    }
}
for(const mode of ['single','best','all']){
    nodes.generation.value=mode;context.generationSettings();
    assert(nodes.searchsettings.disabled && !nodes.standardsettings.disabled);
    assert.equal(nodes.n.disabled,mode==='single');assert(!nodes.chunkframes.disabled);
    const body=request();assert.equal(body.seconds,3.2);assert.equal(body.streaming,true);payloads.push(body);
}
nodes.streaming.value='offline';context.generationSettings();assert(nodes.chunkframes.disabled);
for(const mode of ['clap','adaptive_auto','adaptive_manual']){
    nodes.rewardmode.value=mode;const body=request();payloads.push(body);
    assert.equal(body.reward_mode,mode);
    if(mode==='adaptive_manual'){
        assert.deepEqual(body.reward_plan.terms.map(t=>t.dimension),['vn:AROU','eiv:Anger']);
        assert.equal(body.reward_plan.terms[1].target,null);
    }else assert(!Object.hasOwn(body,'reward_plan'));
}
for(const mode of ['off','location','typed']){
    nodes.burstmode.value=mode;nodes.burstsource.value='manual';nodes.bursttiming.value='clock';
    const body=request();payloads.push(body);assert.equal(body.burst.mode,mode);
    assert.equal(body.burst.targets.length,mode==='off'?0:1);
}
nodes.temperature.value='0';nodes.cfg.checked=false;
let body=request();payloads.push(body);assert.equal(body.temperature,0);assert.equal(body.guidance,1);
nodes.seed.valid=false;
assert.throws(request,/valid value/);assert(nodes.seed.card.open && nodes.seed.focused);
nodes.seed.valid=true;
nodes.reference.value='fixture-reference';context.updateConfigurationSummary();
nodes['voice-choice'].value='custom';context.updateConfigurationSummary();
assert(nodes['configuration-summary'].textContent.includes('custom reference'));
context.smallLoras={selection:()=>({id:'subset_pitch_quality_p2_rank016',weight:1}),summary:()=> 'DPO Rank 16'};
body=request();payloads.push(body);assert.deepEqual(body.adapters,{});
assert.equal(body.voice_mode,'custom');context.updateConfigurationSummary();
assert(!nodes['configuration-summary'].textContent.includes('DPO Rank 16'));
for(const mode of ['ears_mix','ears_only'])for(const model of ['base','medium'])for(const metric of ['ce','pq','cu','pc','mos','ovrl'])for(const weight of ['0','.5','1']){
    nodes['ranking-mode'].value=mode;nodes['ears-model'].value=model;nodes['ears-metric'].value=metric;nodes['ears-weight'].value=weight;
    body=request();payloads.push(body);
    assert.deepEqual(body.ranking,{mode,model,metric,weight:Number(weight)});
    assert.equal(body.reward_mode,'clap');assert.equal(body.burst.mode,'off');assert.equal(body.speaker.method,'off');
}
context.smallEarsAvailable=false;assert.throws(request,/Ears is not ready/);
const result={verification:'Actual production UI functions with DOM fixtures; no browser/CSS or GPU claim',
    search_combinations:64,other_cases:payloads.length-64,payloads};
fs.writeFileSync(path.join(__dirname,'test-results/ui-controller.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
