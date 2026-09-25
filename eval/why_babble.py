"""Which of the suspects actually produces the babble?"""
import sys, json, base64, requests, re, numpy as np
sys.path.insert(0,'/mnt/nvme/moss-15-v2')
import timed_script, config

APP="http://127.0.0.1:8792"
WANT=("Get away from the window, Ellie, and come to me now! There is someone "
      "outside the glass, but we are on the second floor and there is nowhere "
      "for them to stand! Do not turn around again; take my hand, keep your "
      "eyes on me, and run for the hall!")
GEN=("a woman's voice, in their thirties, speaking with Standard American. "
     "intense horror is involuntary and fully unleashed - the opening scream "
     "tears out before thought, then the words come ragged and breath-starved. "
     "genuine and spontaneous, like a real person in a real moment, not acted.")

D="(intensely horrified, letting it out, not hiding it, ragged and protective)"
D2="(still intensely horrified, forcing the words through, breath ragged)"
D3="(extremely horrified, fully unleashed, command collapsing toward panic)"
S1="Get away from the window, [0.3 seconds pause] Ellie, and come to me now!"
S2=("There is someone outside the glass, [0.3 seconds pause] but we are on the "
    "second floor and there is nowhere for them to stand!")
S3=("Do not turn around again; take my hand, keep your eyes on me, and run for "
    "the hall!")

VAR={
 "a_wie_produziert":   f"(scream, 0.7 seconds) {D} (scream) {S1} (fearful gasp, 0.2 seconds) {D2} {S2} (sharp inhale, 0.15 seconds) {D3} {S3}",
 "b_ein_scream":       f"(scream, 0.7 seconds) {D} {S1} (fearful gasp, 0.2 seconds) {D2} {S2} (sharp inhale, 0.15 seconds) {D3} {S3}",
 "c_scream_nach_wort": f"{D} Get away from the window! (scream, 0.7 seconds) [0.3 seconds pause] Ellie, and come to me now! (fearful gasp, 0.2 seconds) {D2} {S2} (sharp inhale, 0.15 seconds) {D3} {S3}",
 "d_ohne_scream":      f"{D} {S1} (fearful gasp, 0.2 seconds) {D2} {S2} (sharp inhale, 0.15 seconds) {D3} {S3}",
 "e_ohne_bursts":      f"{D} {S1} {D2} {S2} {D3} {S3}",
}
BASE=[("sft3_dpo:p2",1.0),("sft3_voice:emolia_c1699",1.0),
      ("sft3_quality:genuineness_high",0.25),("sft3_quality:blend_high",0.5),
      ("sft3_quality:esthetics_high",0.5),("sft3_voicenet:VALN_low",1.5),
      ("sft3_qdpo:quality_dpo",1.5)]
BURST=[("burst_abl:ablation_d2_matched__scream",1.5),("burst_v2:fearful_gasp",1.5)]
EMO_WRONG=[("sft3_emotion:Jealousy_and_Envy",1.0)]
EMO_RIGHT=[("sft3_emotion:Fear",1.0)]

def norm(s):
    return re.sub(r"\s+"," ",re.sub(r"[^\w\s']"," ",(s or "").lower())).strip()
def wer(hyp, ref):
    a,b=norm(hyp).split(),norm(ref).split()
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): d[i][0]=i
    for j in range(len(b)+1): d[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1]))
    return d[len(a)][len(b)]/max(len(b),1)

def run(name, script, loras, seed=1234):
    tg,fr,pl=timed_script.render(script)
    gl=timed_script.general_line(GEN, fr/config.FRAME_RATE, "EN", None)
    r=requests.post(f"{APP}/api/say", json={
        "text":tg,"instruction":f"GENERAL: {gl}\nSCRIPT:\n{tg}","tokens":fr,
        "language":"English","seed":seed,"loras":loras,"align":True,
        "sidon":False}, timeout=600)
    if r.status_code!=200: return name,None,None,f"http {r.status_code}"
    d=r.json(); pcm=base64.b64decode(d["pcm"])
    sr=d.get("sr",48000)
    hdr=(b"RIFF"+ (36+len(pcm)).to_bytes(4,'little') + b"WAVEfmt "+
         (16).to_bytes(4,'little')+(1).to_bytes(2,'little')+(1).to_bytes(2,'little')+
         sr.to_bytes(4,'little')+(sr*2).to_bytes(4,'little')+
         (2).to_bytes(2,'little')+(16).to_bytes(2,'little')+b"data"+
         len(pcm).to_bytes(4,'little'))
    a=requests.post(f"{APP}/api/asr", data=hdr+pcm,
                    headers={"content-type":"application/octet-stream"}, timeout=300)
    txt=a.json().get("text","") if a.status_code==200 else ""
    return name, wer(txt,WANT), len(pcm)/2/sr, txt

print(f"{'variante':<22}{'wer':>7}{'sek':>7}  transkript (anfang)")
for name,sc in VAR.items():
    lor = BASE + (BURST if "scream" in sc or "gasp" in sc or "inhale" in sc else []) + EMO_WRONG
    n,w,sec,txt = run(name, sc, lor)
    print(f"{n:<22}{(w if w is not None else -1):>7.3f}{(sec or 0):>7.1f}  {str(txt)[:95]}")
print()
print("=== emotionsadapter: falsch vs richtig, sonst identisch ===")
for tag,emo in (("Jealousy_and_Envy (wie live)",EMO_WRONG),("Fear (passend)",EMO_RIGHT),("keiner",[])):
    n,w,sec,txt=run(tag, VAR["b_ein_scream"], BASE+BURST+emo)
    print(f"{tag:<30}{(w if w is not None else -1):>7.3f}{(sec or 0):>7.1f}  {str(txt)[:80]}")
