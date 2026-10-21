"""Small deterministic real-time DAC clock for synthetic WebSocket checks.

Chunk arrival is not playback completion. No physical device is simulated;
timestamps/position tags follow the browser wire contract, including gaps.
"""
class SyntheticPlayback:
    def __init__(self,lead16=640):
        self.lead16=lead16;self.until=0;self.segments=[];self.sent={};self.stopped={}
    def add(self,rid,n24,now16):
        if not rid or rid in self.stopped:return
        offset=self.sent.get(rid,0);self.sent[rid]=offset+n24
        start=max(now16+self.lead16,self.until);end=start+round(n24*2/3)
        self.segments.append((start,end,rid,offset,n24));self.until=end
    def pieces(self,start16,n):
        active=[];cuts={0,n};end16=start16+n
        for a,b,rid,p,count in self.segments:
            b=min(b,self.stopped.get(rid,b))
            if b<=start16 or a>=end16:continue
            lo=max(a,start16)-start16;hi=min(b,end16)-start16
            active.append((lo,hi,rid,p+round((start16+lo-a)*1.5)));cuts.update((lo,hi))
        edges=sorted(cuts);parts=[]
        for a,b in zip(edges,edges[1:]):
            value=next(((rid,p+round((a-lo)*1.5)) for lo,hi,rid,p in active if lo<=a<hi),(0,-1))
            parts.append((a,b,*value))
        return parts
    def acknowledgments(self,now16):
        positions={}
        for a,b,rid,p,count in self.segments:
            if rid in self.stopped or now16<a:continue
            positions[rid]=max(positions.get(rid,0),min(p+count,p+round((min(now16,b)-a)*1.5)))
        return positions
    def cancel(self,now16):
        for rid in self.sent:self.stopped[rid]=min(self.stopped.get(rid,now16),now16)
        self.until=now16
