// Small CPU z-buffer / Gouraud renderer. No WebGL, CDN or external dependency.
// Shared pixel-edge rules and depth testing replace unstable polygon sorting.
export class SoftwareSurface{
 constructor(){this.canvas=document.createElement('canvas');this.ctx=this.canvas.getContext('2d');this.width=0;}
 resize(width,height){
  width=Math.max(1,Math.round(width));height=Math.max(1,Math.round(height));
  if(width===this.width&&height===this.height)return;
  this.width=this.canvas.width=width;this.height=this.canvas.height=height;
  this.image=this.ctx.createImageData(width,height);this.bytes=this.image.data;
  this.depth=new Float32Array(width*height);
 }
 begin(){this.bytes.fill(0);this.depth.fill(Infinity);this.triangles=0;}
 triangle(a,b,c,colors,texture=null,uv=null){
  const width=this.width,height=this.height;
  const area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  if(Math.abs(area)<1e-8)return;
  const loX=Math.max(0,Math.floor(Math.min(a[0],b[0],c[0]))),hiX=Math.min(width-1,Math.ceil(Math.max(a[0],b[0],c[0])));
  const loY=Math.max(0,Math.floor(Math.min(a[1],b[1],c[1]))),hiY=Math.min(height-1,Math.ceil(Math.max(a[1],b[1],c[1])));
  if(hiX<loX||hiY<loY)return;this.triangles++;
  const inv=1/area,dx0=(b[1]-c[1])*inv,dy0=(c[0]-b[0])*inv,dx1=(c[1]-a[1])*inv,dy1=(a[0]-c[0])*inv;
  const x=loX+.5,y=loY+.5;
  let row0=((b[0]-x)*(c[1]-y)-(b[1]-y)*(c[0]-x))*inv;
  let row1=((c[0]-x)*(a[1]-y)-(c[1]-y)*(a[0]-x))*inv;
  const ia=1/a[2],ib=1/b[2],ic=1/c[2],pixels=this.bytes,depths=this.depth;
  for(let py=loY;py<=hiY;py++,row0+=dy0,row1+=dy1){
   let w0=row0,w1=row1,index=py*width+loX;
   for(let px=loX;px<=hiX;px++,w0+=dx0,w1+=dx1,index++){
    const w2=1-w0-w1;if(w0< -1e-7||w1< -1e-7||w2< -1e-7)continue;
    const iz=w0*ia+w1*ib+w2*ic,z=1/iz;if(z>=depths[index])continue;
    const offset=index*4;
    if(texture){
     const u=(w0*uv[0][0]*ia+w1*uv[1][0]*ib+w2*uv[2][0]*ic)*z;
     const v=(w0*uv[0][1]*ia+w1*uv[1][1]*ib+w2*uv[2][1]*ic)*z;
     const tx=Math.max(0,Math.min(texture.width-1.001,u)),ty=Math.max(0,Math.min(texture.height-1.001,v));
     const ix=tx|0,iy=ty|0,fx=tx-ix,fy=ty-iy,t=(iy*texture.width+ix)*4,tw=texture.width*4,data=texture.data;
     const k0=(1-fx)*(1-fy),k1=fx*(1-fy),k2=(1-fx)*fy,k3=fx*fy;
     const alpha=data[t+3]*k0+data[t+7]*k1+data[t+tw+3]*k2+data[t+tw+7]*k3;
     if(alpha<8)continue;
     for(let channel=0;channel<3;channel++)pixels[offset+channel]=data[t+channel]*k0+data[t+4+channel]*k1+data[t+tw+channel]*k2+data[t+tw+4+channel]*k3;
     pixels[offset+3]=alpha;
    }else{
     for(let channel=0;channel<3;channel++)pixels[offset+channel]=w0*colors[0][channel]+w1*colors[1][channel]+w2*colors[2][channel];
     pixels[offset+3]=255;
    }
    depths[index]=z;
   }
  }
 }
 finish(){this.ctx.putImageData(this.image,0,0);return this.canvas;}
}
