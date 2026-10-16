/* Checkpoint switching is serialized by the server, not a second GPU process. */
(() => {
  const $=id=>document.getElementById(id);
  const releases={small_rl:{label:'Small 1.1 Alpha RL · r128 v1 · Default',max:8,search:8,rl:true},
                  large_rl:{label:'Large 1.1 Alpha RL · r128 v1',max:2,search:2,rl:true},
                  large_acting:{label:'Large Acting · Step 432',max:4,search:4},
                  small_acting:{label:'Small · 12 Oct 2026',max:8,search:8},
                  small_alpha:{label:'Small 1.1 Alpha · Step 3,775 · Experimental',max:8,search:8,experimental:true},
                  large_alpha:{label:'Large 1.1 Alpha · Step 2,377 · Experimental',max:4,search:4,experimental:true}};
  let switching=false;
  function limits(){
    const release=releases[$('model-choice').value];
    $('model-batch').max=String(release.max);
    $('model-batch').value=String(Math.min(Number($('model-batch').value)||1,release.max));
    for(const option of $('searchn').options)option.disabled=Number(option.value)>release.search;
    if(Number($('searchn').value)>release.search)$('searchn').value=String(release.search);
    $('speaker-steering').checked=false;$('speaker-steering').disabled=true;
    $('speaker-steering-status').textContent='Off: legacy Layer-16 vectors are incompatible with Acting checkpoints.';
    $('model-help').textContent=release.label+' · Maximum parallel batch '+release.max+'. Best-of-N runs candidates in parallel batches automatically. Live search: up to '+release.search+' paths. '+(release.rl?'Verified base + fixed RL adapter; FP32 policy, TF32 off. Recommended temperature: 0.70. First use of a new model/batch may take several minutes to compile. Large RL uses CPU ASR/Ears/alignment to fit 24 GB VRAM; scoring and buffered playback are slower. ':'')+'Legacy external adapters and layer vectors stay off.';
    $('model-experiment-note').hidden=!release.experimental;
    globalThis.actingLabel=release.label;
    $('checkpoint-summary').textContent=release.label+(release.rl?' · verified base + RL adapter · FP32':' · complete checkpoint');
    globalThis.generationSettings?.();
  }
  globalThis.actingModelState=s=>{
    // Display the model actually resident on the server, including reloads
    // after a model switch and changes made from another browser.
    if(s.model_id && !s.busy && !switching)$('model-choice').value=s.model_id;
    if(switching)ready=false; // A stale poll must not re-enable generation.
    $('model-choice').disabled=switching||!!s.busy||!s.ready;
    $('model-batch').disabled=switching||!!s.busy||!s.ready;
    limits();
    const cache=s.checkpoint_ram?.models||{};
    for(const id of ['small_alpha','large_alpha','large_rl']){
      const option=Array.from($('model-choice').options).find(o=>o.value===id);
      // Keep unfinished preloads visible, but do not start a long pending
      // selection while its CPU cache is being prepared.
      option.disabled=!cache[id]||cache[id].status==='loading';
    }
    $('model-ram-status').textContent=['small_rl','large_rl','small_alpha','large_alpha'].map(id=>{
      const row=cache[id];return releases[id].label.split(' · Experimental')[0]+': '+
        (row?.status==='ready'?(row.bytes/1e9).toFixed(2)+' GB in CPU RAM':row?.status==='error'?'preload failed (retry on selection)':row?.status==='loading'?'preloading CPU RAM…':'waiting for RAM preload');
    }).join(' · ');
  };
  $('dynamic-reference').onchange=()=>{voiceHelp();updateConfigurationSummary();};
  $('model-choice').onchange=async()=>{
    switching=true;ready=false;
    limits();$('model-choice').disabled=true;$('send').disabled=true;
    $('busy').textContent='Loading Acting checkpoint and warming streaming…';
    try{
      await checked(await fetch('/api/models/select',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({model:$('model-choice').value})}));
      $('busy').textContent='Model ready: '+releases[$('model-choice').value].label;
    }catch(error){$('busy').textContent=error.message;}
    finally{switching=false;await state();}
  };
  limits();
})();
