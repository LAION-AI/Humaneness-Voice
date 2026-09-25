"""Is guidance 3 better than no guidance? Same script, one number different."""
import sys, json, base64, re, asyncio, statistics as st, math, os
import httpx, numpy as np
sys.path.insert(0,'/mnt/nvme/moss-15-v2'); sys.path.insert(0,'/mnt/nvme/arena')
import timed_script, config, harness
APP="http://127.0.0.1:8792"
OUT="/mnt/nvme/arena/audio/cfgab34"; os.makedirs(OUT, exist_ok=True)
GVALS=[1.0,3.0,4.0]
SEEDS=[101,202,303]
# The always-on part of a shipped turn.  Measuring guidance on a bare model
# would answer a question nobody has: every take in the demo carries these.
STACK=[("sft3_dpo:p2",1.0),("sft3_voice:emolia_c1699",1.0),
       ("sft3_quality:genuineness_high",0.25),("sft3_quality:blend_high",0.5),
       ("sft3_quality:esthetics_high",0.5),("sft3_qdpo:quality_dpo",1.5)]

TASKS=[
 ("EN-fear","English","Face at the Window",
  "During a blackout, a parent sees an unknown face at the second-floor nursery window.",
  "Begin with an involuntary scream, then force protective words through extreme horror.",
  "Get away from the window, Ellie, and come to me now! There is someone outside the glass, but we are on the second floor. Take my hand and run for the hall!"),
 ("EN-joy","English","The Elevator Tune",
  "A coworker discovers the elevator plays their old voicemail greeting as music.",
  "Keep the amusement tucked behind a courteous office voice.",
  "The elevator is playing your old voicemail greeting between floors. It says, leave a message, and then the doors open. I promise I will not tell anyone until after lunch."),
 ("DE-grief","German","Der letzte Anruf",
  "Jemand hoert die alte Mailbox-Ansage eines verstorbenen Freundes wieder ab.",
  "Trauer, die zurueckgehalten wird und nur an den Satzraendern durchbricht.",
  "Ich habe seine Nummer noch einmal gewaehlt, nur um die Ansage zu hoeren. Seine Stimme klingt genau wie frueher, als waere nichts passiert. Ich weiss, dass ich das nicht mehr tun sollte."),
 ("DE-anger","German","Nicht in meinem Kurs",
  "Eine Lehrerin stellt einen Schueler zur Rede, der die Praesentation einer Mitschuelerin geloescht hat.",
  "Zorn, der beherrscht wird, mit wachsender Schaerfe.",
  "Sag mir jetzt, was mit Linas Datei passiert ist, bevor jemand anderes etwas sagt. Du hast eine Gelegenheit, das in Ordnung zu bringen. Findest du das komisch, erklaerst du es dem Direktor."),
]

async def script_for(cli, t):
    key,lang,title,ctx,direction,text = t
    msg = json.dumps({"track":"emotion","target":{"label":"","intensity":"intense"},
        "instruction":{"context":ctx,"performance_direction":direction},
        "script":{"text":text}}, ensure_ascii=False)
    r = await cli.post(f"{APP}/api/turn", json={"message":msg,"session":f"ab-{key}",
        "best_of":1,"sidon":False}, timeout=900)
    ev,i,raw=[],0,r.content
    import struct
    while i<len(raw):
        tt=raw[i];n=struct.unpack(">I",raw[i+1:i+5])[0];p=raw[i+5:i+5+n];i+=5+n
        if tt==0: ev.append(json.loads(p))
    llm=next((e for e in ev if e.get("type")=="llm"),{})
    return llm.get("script"), llm.get("general"), llm.get("general_unc"), llm.get("language") or lang

async def takes(cli, script, general, gen_unc, lang, g):
    tg,fr,pl = timed_script.render(script)
    lc = "DE" if str(lang).lower().startswith(("ger","de")) else "EN"
    gl = timed_script.general_line(general, fr/config.FRAME_RATE, lc)
    item={"text":tg,"tokens":fr,"language":lang,
          "instruction":f"GENERAL: {gl}\nSCRIPT:\n{tg}"}
    if g>1.001:
        tu=timed_script.neutralise(tg)
        gu=timed_script.general_line(gen_unc or general, fr/config.FRAME_RATE, lc)
        item["instruction_unc"]=f"GENERAL: {gu}\nSCRIPT:\n{tu}"; item["text_unc"]=tu
    out=[]
    for seed in SEEDS:
        r=await cli.post(f"{APP}/api/say_batch", json={"items":[dict(item)],
            "guidance":g,"seed":seed,"loras":STACK}, timeout=900)
        d=r.json()
        pcm=base64.b64decode(d["pcm"][0]); sr=d.get("sr",48000)
        hdr=(b"RIFF"+(36+len(pcm)).to_bytes(4,'little')+b"WAVEfmt "+(16).to_bytes(4,'little')+
             (1).to_bytes(2,'little')+(1).to_bytes(2,'little')+sr.to_bytes(4,'little')+
             (sr*2).to_bytes(4,'little')+(2).to_bytes(2,'little')+(16).to_bytes(2,'little')+
             b"data"+len(pcm).to_bytes(4,'little'))
        out.append(hdr+pcm)
    return out

async def main():
    cli=httpx.AsyncClient(timeout=900)
    rows=[]
    for t in TASKS:
        key,lang,title,ctx,direction,text=t
        sc,gen,gunc,lg = await script_for(cli,t)
        if not sc: print(key,"kein skript"); continue
        print(f"{key}: {sc[:90]}...", flush=True)
        for g in GVALS:
            wavs = await takes(cli, sc, gen, gunc, lg, g)
            for k,w in enumerate(wavs):
                fn=f"{OUT}/{key}_g{g:g}_{k}.wav"; open(fn,'wb').write(w)
                rows.append({"task":key,"lang":lang,"g":g,"seed":SEEDS[k],"file":fn,
                             "ctx":ctx,"dir":direction,"text":text})
        print(f"  {key}: {len(SEEDS)*2} takes", flush=True)
    # bewerten
    import types
    R=types.SimpleNamespace(http=cli,key=config.luna_key(),
                            sem_judge=asyncio.Semaphore(8),
                            log=lambda m: print(m,flush=True),
                            audio=OUT, a=types.SimpleNamespace())
    async def judge(row):
        task={"target":{"label":"","intensity":"intense"},
              "instruction":{"context":row["ctx"],"performance_direction":row["dir"]},
              "script":{"text":row["text"]}}
        r=dict(row); r["audio"]=os.path.basename(row["file"])
        return await harness.Run.judge(R, r, task)
    res=await asyncio.gather(*[judge(r) for r in rows])
    for r,s in zip(rows,res): r["scores"]=s
    json.dump(rows, open("/mnt/nvme/arena/runs/cfg_ab34.json","w"), ensure_ascii=False, indent=1)
    ok=[r for r in rows if r.get("scores")]
    print(f"\n{len(ok)} clips bewertet\n")
    print(f"{'':<12}{'pleasant':>10}{'fit':>8}{'natural':>9}{'total':>8}")
    for g in GVALS:
        sub=[r for r in ok if r["g"]==g]
        if not sub: continue
        print(f"g={g:<10g}" + "".join(
            f"{st.mean([x['scores'][k]['score'] for x in sub]):>10.2f}" for k in harness.RUBRICS)
            + f"{st.mean([x['scores']['total'] for x in sub]):>8.2f}")
    # gepaart je (task, seed)
    def paired(ga, gb, sub=None):
        d=[]
        pool = sub if sub is not None else ok
        for t in {r["task"] for r in pool}:
            for sd_ in SEEDS:
                a=[r for r in pool if r["task"]==t and r["seed"]==sd_ and r["g"]==ga]
                b=[r for r in pool if r["task"]==t and r["seed"]==sd_ and r["g"]==gb]
                if a and b: d.append(a[0]["scores"]["total"]-b[0]["scores"]["total"])
        if len(d)<3: return None
        m=st.mean(d); sd2=st.stdev(d); tt=m/(sd2/len(d)**0.5) if sd2 else 0
        return m, tt, math.erfc(abs(tt)/math.sqrt(2)), len(d), sum(1 for x in d if x>0)
    print()
    for ga,gb in ((3.0,1.0),(4.0,1.0),(4.0,3.0)):
        r=paired(ga,gb)
        if r: print(f"gepaart (g{ga:g} - g{gb:g}): {r[0]:+.2f} punkte, t {r[1]:+.2f}, "
                    f"p {r[2]:.3f}, n={r[3]}, besser in {r[4]}/{r[3]}")
    print()
    for lang in ("English","German"):
        sub=[r for r in ok if r["lang"]==lang]
        for ga,gb in ((3.0,1.0),(4.0,3.0)):
            r=paired(ga,gb,sub)
            if r: print(f"  {lang:<9} g{ga:g}-g{gb:g} {r[0]:+.2f} (n={r[3]})")
asyncio.run(main())
