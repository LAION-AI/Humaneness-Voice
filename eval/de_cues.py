"""German line: do German brackets break it, and does the wrong emotion?"""
import sys, statistics as st, requests, base64, re
sys.path.insert(0,'/mnt/nvme/moss-15-v2')
import timed_script, config
APP="http://127.0.0.1:8792"
WANT=("Geh weg vom Fenster, Ellie, und komm jetzt zu mir! Da ist jemand draußen "
      "vor der Scheibe, aber wir sind im zweiten Stock, und dort kann niemand "
      "stehen... Dreh dich nicht noch einmal um; nimm meine Hand, sieh mich an, "
      "und lauf in den Flur!")
S1="Geh weg vom Fenster, [0.3 seconds pause] Ellie, und komm jetzt zu mir!"
S2=("Da ist jemand draußen vor der Scheibe, [0.3 seconds pause] aber wir sind im "
    "zweiten Stock, und dort kann niemand stehen...")
S3="Dreh dich nicht noch einmal um; nimm meine Hand, sieh mich an, und lauf in den Flur!"
DE=["(überwältigend verängstigt, völlig ungehemmt, die Stimme bricht am Rand eines weiteren Schreis)",
    "(weiterhin intensiv entsetzt, die Worte durch die Panik gezwungen)",
    "(extrem beschützend und panisch, die Dringlichkeit übernimmt)"]
EN=["(overwhelmingly terrified, letting it out, not hiding it, the voice breaking at the edge of another scream)",
    "(still intensely horrified, the words forced through panic)",
    "(extremely protective and panicked, urgency taking over)"]
def script(cues):
    return (f"(scream, 0.8 seconds) {cues[0]} {S1} (ragged breath, 0.3 seconds) "
            f"{cues[1]} {S2} (fearful gasp, 0.2 seconds) {cues[2]} {S3}")
G_DE=("a woman's voice, in their thirties, speaking with Standard American. "
      "intensiv von überwältigender Angst erfasst und völlig ungehemmt — der erste "
      "Schrei bricht unfreiwillig, blutcurdling und scharf hervor; danach kommen "
      "die Worte atemlos, rau und am Rand eines weiteren Schreis.")
G_EN=("a woman's voice, in their thirties, speaking with Standard American. "
      "intense terror, involuntary and fully unleashed - the first scream tears "
      "out blood-curdling and sharp, then the words come breathless and ragged "
      "at the edge of another scream.")
BASE=[("sft3_dpo:p2",1.0),("sft3_voice:emolia_c1699",1.0),
      ("sft3_quality:genuineness_high",0.25),("sft3_quality:blend_high",0.5),
      ("sft3_quality:esthetics_high",0.5),("sft3_voicenet:TENS_high",1.5),
      ("sft3_qdpo:quality_dpo",1.5)]
# sum 2.0, wie die neue budgetregel es liefert
BURST=[("burst_abl:ablation_d2_matched__scream",0.8),
       ("burst_v2:fearful_gasp",0.6),("burst:ragged_breath",0.6)]
def norm(s): return re.sub(r"\s+"," ",re.sub(r"[^\w\s'äöüßÄÖÜ]"," ",(s or "").lower())).strip()
def wer(h,r):
    a,b=norm(h).split(),norm(r).split()
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): d[i][0]=i
    for j in range(len(b)+1): d[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
    return d[len(a)][len(b)]/max(len(b),1)
def run(sc,gen,lor,seed,reads=None):
    tg,fr,pl=timed_script.render(sc)
    gl=timed_script.general_line(gen, fr/config.FRAME_RATE, "DE", reads)
    r=requests.post(f"{APP}/api/say", json={"text":tg,
        "instruction":f"GENERAL: {gl}\nSCRIPT:\n{tg}","tokens":fr,
        "language":"German","seed":seed,"loras":lor,"align":True,"sidon":False},
        timeout=600)
    if r.status_code!=200: return None
    d=r.json(); pcm=base64.b64decode(d["pcm"]); sr=d.get("sr",48000)
    hdr=(b"RIFF"+(36+len(pcm)).to_bytes(4,'little')+b"WAVEfmt "+(16).to_bytes(4,'little')+
         (1).to_bytes(2,'little')+(1).to_bytes(2,'little')+sr.to_bytes(4,'little')+
         (sr*2).to_bytes(4,'little')+(2).to_bytes(2,'little')+(16).to_bytes(2,'little')+
         b"data"+len(pcm).to_bytes(4,'little'))
    a=requests.post(f"{APP}/api/asr", data=hdr+pcm,
                    headers={"content-type":"application/octet-stream"}, timeout=300)
    return wer(a.json().get("text","") if a.status_code==200 else "", WANT)
SEEDS=[11,22,33,44]
conds=[
 ("deutsche cues + deutsches GENERAL", script(DE), G_DE, BASE+BURST, None),
 ("ENGLISCHE cues + deutsches GENERAL", script(EN), G_DE, BASE+BURST, None),
 ("englische cues + ENGLISCHES GENERAL", script(EN), G_EN, BASE+BURST, None),
 ("deutsche cues + englisches GENERAL", script(DE), G_EN, BASE+BURST, None),
 ("englisch/englisch + 'reads as jealousy and envy'", script(EN), G_EN, BASE+BURST, "Jealousy_and_Envy"),
 ("englisch/englisch + falscher emo-adapter", script(EN), G_EN,
  BASE+BURST+[("sft3_emotion:Jealousy_and_Envy",1.0)], None),
 ("englisch/englisch + Fear-adapter", script(EN), G_EN,
  BASE+BURST+[("sft3_emotion:Fear",1.0)], None),
]
print(f"{'bedingung':<52}" + "".join(f"{s:>7}" for s in SEEDS) + f"{'median':>8}{'kaputt':>8}")
for tag,sc,gen,lor,reads in conds:
    ws=[run(sc,gen,lor,s,reads) for s in SEEDS]
    ws=[w if w is not None else float('nan') for w in ws]
    bad=sum(1 for w in ws if w and w>0.25)
    print(f"{tag:<52}"+"".join(f"{w:>7.2f}" for w in ws)+
          f"{st.median(ws):>8.3f}{bad:>6}/{len(SEEDS)}", flush=True)
