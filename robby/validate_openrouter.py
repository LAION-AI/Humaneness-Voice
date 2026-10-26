"""Actual allowlisted model routes, streaming replies, tools and secret boundary."""
import asyncio,json,stat,time
from pathlib import Path
import httpx
from credential_store import store
from conversation_director import CompactDirector
from llm_backends import config,refresh_providers

ROOT=Path(__file__).resolve().parent
async def main():
    report={'passed':False,'started':time.time(),'rows':[],'physical_microphone':False}
    vault=store();secret=vault.owner()
    assert secret.encode() not in (vault.root/'openrouter.enc').read_bytes()
    assert all(stat.S_IMODE((vault.root/name).stat().st_mode)==0o600 for name in ('openrouter.enc','master.key'))
    report['owner_encrypted_outside_source']=not vault.root.resolve().is_relative_to(ROOT.resolve())
    async with httpx.AsyncClient(timeout=90) as c:
        await refresh_providers(c)
        response=await c.get('http://127.0.0.1:8798/api/llm/models');response.raise_for_status()
        assert secret not in response.text and 'Authorization' not in response.text
        available=response.json();assert available['server_key_available']
        for model,provider,text in [('qwen','cerebras/fp16','Sag mir freundlich guten Morgen.'),
                ('qwen','cerebras/fp16','Schreibe ein lustiges Gedicht mit vier Zeilen.'),
                ('local','local','Schlage ein Spiel vor.'),
                ('qwen','deepinfra/bf16','Sag nur: Guten Morgen.')]:
            start=time.perf_counter();r=await c.post('http://127.0.0.1:8798/api/bolt/llm',json={'text':text,'llm_model':model,'llm_provider':provider});r.raise_for_status()
            assert secret not in r.text
            parts=[];done=None
            for block in r.text.split('\n\n'):
                if not block.startswith('event: '):continue
                kind=block.splitlines()[0][7:];data=json.loads(next(x[6:] for x in block.splitlines() if x.startswith('data: ')))
                assert kind!='error',data
                if kind=='sentence':parts.append(data['text'])
                if kind=='done':done=data
            assert parts and done and done['llm_provider']==provider,(model,provider,parts,done)
            if 'vier Zeilen' in text:assert len(parts)==4,parts
            row={'model':model,'provider':provider,'actual_provider':done.get('llm_actual_provider','local'),
                'input':text,'reply':'\n'.join(parts),'first_token_ms':done.get('llm_ttft_ms'),
                'first_sentence_ms':done.get('llm_first_sentence_ms'),'total_http_ms':round((time.perf_counter()-start)*1000,2),'passed':True}
            report['rows'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
        result=await CompactDirector().generate(c,'Wenn ich über Bananen spreche, unterbrich mich und sage Stopp, stopp!',{},[],
            interaction_state={'enabled':True,'rules':[]},provider=config({'model':'qwen','provider':'cerebras/fp16'}))
        assert any(a['name']=='set_interruption_rule' for a in result.get('interaction_actions',[])),result['text']
        report['real_rule_tool']={'passed':True,'reply':result['text'],'actions':result['interaction_actions'],'provider':result.get('llm_actual_provider')}
    report.update(passed=True,finished=time.time(),key_never_in_responses=True)
    encoded=json.dumps(report,indent=2,ensure_ascii=False);assert secret not in encoded
    (ROOT/'static/openrouter-check.json').write_text(encoded);print('PASSED',report['passed'],flush=True)
if __name__=='__main__':asyncio.run(main())
