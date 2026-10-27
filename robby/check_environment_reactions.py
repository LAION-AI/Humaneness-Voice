"""Actual tiny-model reactions, including a memory question with a sound hint."""
import asyncio,json
from pathlib import Path
import httpx
from quick_reaction import generate,supported_sound_reaction,word_count

async def main():
    rows=[]
    async with httpx.AsyncClient(timeout=5,trust_env=False) as client:
        for caption in ('A single, sharp, distinct percussive sound with a dry, crisp quality.',
                        'A door appears to close with a loud slam.','A short electronic beep is audible.'):
            for repeat in range(3):
                env={'kind':'sound','caption':caption}
                result=await generate(client,'',{},[],'German',environment=env)
                rows.append({'caption':caption,'repeat':repeat,'result':result})
                if result['text']:
                    assert word_count(result['text'])<=6 and supported_sound_reaction(result['text'],env)
        result=await generate(client,'Wie heißt meine Freundin?',{'display_name':'Bob'},[],
            memory={'entries':[{'title':'Bob','kind':'person','summary':'Bobs Freundin heißt Alice.','detail':'Bob hat gesagt: Meine Freundin heißt Alice.'}]},
            environment={'kind':'sound','caption':'A short knock was heard.'})
        rows.append({'user':'Wie heißt meine Freundin?','result':result,'memory_and_sound':True})
        assert result.get('text') and 'alice' in result['text'].lower(),result
    out=Path(__file__).resolve().parent/'static/environment-reactions.json'
    out.write_text(json.dumps(rows,ensure_ascii=False,separators=(',',':')))
    print(json.dumps({'rows':len(rows),'valid':sum(bool(r['result']['text']) for r in rows),'texts':[r['result']['text'] for r in rows]},ensure_ascii=False))

if __name__=='__main__':asyncio.run(main())
