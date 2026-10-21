// Timestamp mapping, independent of network arrival time. Apache-2.0 integration.
export class PlaybackClock {
  constructor(context){this.context=context;this.segments=[];this.inputLatency=null;this.acousticLatency=0;this.delay=0;this.method='unavailable';this.missing=[];}
  observe(now=performance.now()){
    const c=this.context,t=c.getOutputTimestamp?.();
    const plausible=t&&t.contextTime>0&&t.performanceTime>0&&Math.abs(now-t.performanceTime)<1500;
    const presented=plausible?t.contextTime+(now-t.performanceTime)/1000:NaN;
    const measured=c.currentTime-presented;
    if(plausible&&measured>=-0.015&&measured<1.5){this.delay=Math.max(0,measured);this.method='output timestamp';}
    else{this.delay=Math.max(0,(c.baseLatency||0)+(c.outputLatency||0));this.method='reported latency fallback';}
    this.missing=[];
    if(this.inputLatency===null)this.missing.push('input latency');
    if(!plausible&&!c.outputLatency)this.missing.push('output timestamp/latency');
    return this.delay;
  }
  outputHead(now=performance.now()){this.observe(now);return this.context.currentTime-this.delay;}
  position(now=performance.now()){
    const head=this.outputHead(now),s=this.segments.find(v=>v.t0<=head&&head<v.t1);
    return s?{id:s.id,sample:s.first+Math.floor((head-s.t0)*24000)}:null;
  }
  heardTime(captureTime,now=performance.now()){
    this.observe(now);return captureTime-this.delay-(this.inputLatency||0)-this.acousticLatency;
  }
  add(id,t0,count,first){this.segments.push({id,t0,t1:t0+count/24000,first});}
  tag(captureTime,now=performance.now()){
    const heard=this.heardTime(captureTime,now);
    const segment=this.segments.find(s=>s.t0<=heard&&heard<s.t1);
    this.segments=this.segments.filter(s=>s.t1>this.context.currentTime-12);
    return segment?[segment.id,segment.first+Math.floor((heard-segment.t0)*24000)]:[0,-1];
  }
  pieces(captureTime,count,rate=16000,now=performance.now()){
    const heard=this.heardTime(captureTime,now),cuts=new Set([0,count]);
    for(const s of this.segments)for(const edge of [s.t0,s.t1]){
      const index=Math.ceil((edge-heard)*rate-1e-7);if(index>0&&index<count)cuts.add(index);
    }
    const positions=[...cuts].sort((a,b)=>a-b),out=[];
    for(let j=0;j<positions.length-1;j++){
      const first=positions[j],end=positions[j+1],at=heard+first/rate;
      const s=this.segments.find(v=>v.t0<=at+1e-9&&at<v.t1-1e-9);
      out.push({first,end,id:s?s.id:0,at:s?s.first+Math.max(0,Math.floor((at-s.t0)*24000+1e-7)):-1});
    }
    this.segments=this.segments.filter(s=>s.t1>this.context.currentTime-12);
    return out;
  }
  stop(at=this.context.currentTime){
    // Keep already played history: delayed mic blocks can still refer to it.
    for(const s of this.segments)s.t1=Math.min(s.t1,at);
    this.segments=this.segments.filter(s=>s.t1>s.t0);
  }
  report(){return {output_ms:this.delay*1000,input_ms:this.inputLatency===null?null:this.inputLatency*1000,
    method:this.method,uncertain:this.missing.length>0,missing:this.missing};}
}
